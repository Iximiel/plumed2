# Coordination with arrayFire

`FIRECOORDINATION` and `FIRECOORDINATIONFLOAT` depend on [CCCL](https://github.com/NVIDIA/cccl) which is automatically fetched by the cuda compiler (if you use nvcc, you have access to the CCCL headers).

The "inspiration" for this new coordination implementation is the already present fire implementation of SAXS.

As now I did not prepare a lesson in the [school](https://plumed-school.github.io).
So her there is a simple tutorial:

## Implementation

For using _ArrayFire_ you must shift your way of reasoning in up to 4D tensors.

The hands on implementation will consist in three different parts:
 - the data preparation (that depend on the kind of calculation)
 - the calculation (that is the same procedure for any calculation)
 - the output preparation  (that depend on the kind of calculation)

With "kind of calcculation" I mean "two groups" "self" or "pair", that depend on the input string of the cv.

### Implementation: prepare the data

First of all we want to copy to the atomic positions from the cpu to the arrayfire device.

```c++
auto posA = setPositions<float>(&getPositions()[0][0], atomsInA);
auto posB = setPositions<float>(&getPositions()[atomsInA][0], atomsInB);
```
`setPositions` wraps the constructor of `af::array` and a conversion to float (or no conversion if called with the `<double>` arguments), and returns a `(3,second_argument,1,1)` tensor, where the first dimension is the x/y/z coordinate and the "colums" represent the atom whose coordinate is that.

The data in the two position container can visualized like this:

posA:
```
x_A0 x_A1 x_A2 ...
y_A0 y_A1 y_A2 ...
z_A0 z_A1 z_A2 ...
```
posB:
```
x_B0 x_B1 x_B2 ...
y_B0 y_B1 y_B2 ...
z_B0 z_B1 z_B2 ...
```
And `diff = posB - posA`  will store the following data:
```
x_A0-x_B0 x_A1-x_B1 x_A2-x_B2 ...
y_A0-y_B0 y_A1-y_B1 y_A2-y_B2 ...
z_A0-z_B0 z_A1-z_B1 z_A2-z_B2 ...
```
This is how the data is passed in "PAIR" mode

Without the `PAIR` keyword, starting from `posA` and `posB` we'll trade some space for speed up the calculations.

So we use `af::tile` to tile the tensor with the data contained in the positions arrays.

```c++
posB = af::tile(posB,1,1,atomsInA);
```
making `posB` a `(3,atomsInB,atomsInA,1)` tensor,
And the __row 0__ of the first dimension will became:
```
x_B0 x_B1 x_B2 ...
x_B0 x_B1 x_B2 ...
x_B0 x_B1 x_B2 ...
...
```
Whereas for `posA` we change ho AF interprets the tensor with `af::moddims` making it a `(3,1,atomsInA,1)` tensor, and then we tile it to a `(3,atomsInB,atomsInA,1)` tensor

```c++
posA = af::tile(af::moddims(posA,3,1,atomsInA),1,atomsInB,1);
```
And the __row 0__ of the first dimension will became:
```
x_A0 x_A0 x_A0 ...
x_A1 x_A1 x_A1 ...
x_A2 x_A2 x_A2 ...
...
```
so that diff will look like:
```
x_B0-x_A0 x_B1-x_A0 x_B2-x_A0 ...
x_B0-x_A1 x_B1-x_A1 x_B2-x_A1 ...
x_B0-x_A2 x_B1-x_A2 x_B2-x_A2 ...
...
```
and will have dimension `(3,atomsInB,atomsInA,1)`, and can be used in the [next paragraph](#implementation-prepare-the-data), without changing the code.

For the non pair implementation with a single group the procedure is the same, except for

```c++
auto posA = setPositions<float>(&getPositions()[0][0], atomsInA);
auto posB = setPositions<float>(&getPositions()[0][0], atomsInA);
```
And using atomsInA in place of atomsInB


## Implementation: coordination

The distance can simply calculated by subtracting the two arrays positions array (see below how we got `posA` and `posB`):
```c++
auto diff = posB - posA;
```
and then we calculate the square of the distance
```c++
auto ddistSQ = af::sum(diff * diff,0);
```
The second argument will make sure that the distances will be collapsed in a `(1,atomsInB,atomsInA,1)` tensor, because we are reducing on the first dimension.
and then we can proceed as usual, by passing the squared distance through a switching function with a similar implementation to the non accelerated one (see below)
```c++
auto [res, dfunc] = switching(ddistSQ, switchingParameters);
```
Here we are using `auto [variablenames]` because `switching` returns a tuple and we want the compiler to demangle it for us, so no need to use first,second etc...
now we proceed by calculationg the derivatives and the box derivatives:
```c++
auto keys = (ddistSQ < switchingParameters.dmaxSQ) - trueindexes;
auto AFderiv = af::select(keys, dfunc * diff, 0.0);

const unsigned natA = diff.dims()[2];
const unsigned natB = diff.dims()[1];

auto AFvirial=af::array(9,natB,natA,getType<calculateFloat>());
AFvirial.row(0) = AFderiv.row(0) * diff.row(0);
AFvirial.row(1) = AFderiv.row(0) * diff.row(1);
AFvirial.row(2) = AFderiv.row(0) * diff.row(2);
AFvirial.row(3) = AFderiv.row(1) * diff.row(0);
AFvirial.row(4) = AFderiv.row(1) * diff.row(1);
AFvirial.row(5) = AFderiv.row(1) * diff.row(2);
AFvirial.row(6) = AFderiv.row(2) * diff.row(0);
AFvirial.row(7) = AFderiv.row(2) * diff.row(1);
AFvirial.row(8) = AFderiv.row(2) * diff.row(2);
```
See below how we got `trueindexes`, that is used to discard the calculations that are within atoms with the same index. `keys` merge the "discard on indexes" with the" discard on cut off", and it is used with `af::select` to put zeros in those place (because a `+0` does not change a result).

In the previous snippet we are calculating the derivative and the virial with "simple" multiplications.

As before, the first coordinate of the tensor is the "accumulated coordinate", I'm working with this schema becasue it is the way of ho AF is storing the data.

Then we may proceed with the accumulations:

```c++
AFvirial = -af::sum(af::sum(AFvirial, 2),1);

calculateFloat coord;
af::sum(af::sum(af::select(keys,res,0.0),2),1).host(&coord);

```

Here we are summing on dimension 2 and 1 so that we cover all the cases (PAIR only needs one dimension, the other two implementations need 2)

And this concludes the common part. The derivatives shall be accumulated differently for each different implementation.

### Implementation: collect results

If we put the [previous paragraph](#implementation-prepare-the-data) in a function we can return three things:
```c++
auto [
      coordination,
      AFderivative,
      AFvirial
    ] = work(diff, trueindexes, myPBC);
```

 - coordinaton, __alreeady reduced__*
 - the virial/box derivative, __already reduced__*
 - the atoms derivatives, __not reduced__

The derivatives need a slightly different recipe for each of the three implementation, and in hte case of the single group coordination we need to slightly massage also the virial and the coordination

#### PAIR

```c++
//we let c++ take care of the conversion fo a single scalar
coordinationPlumed=coordination;
//ad-hoc written helpers functions
getToHost<float>(AFvirial, DataInterface(virial));
getToHost<float>(-AFderivative, DataInterface(derivativeA));
getToHost<float>( AFderivative, DataInterface(derivativeB));
```
The derivatives are calculated per single pair and do not need any reduction
#### GROUPA, GROUPB

```c++
coordinationPlumed=coordination;
getToHost<calculateFloat>(AFvirial, DataInterface(virial));
getToHost<calculateFloat>(-af::sum(AFderivative,1), DataInterface(derivativeA));
getToHost<calculateFloat>( af::sum(AFderivative,2), DataInterface(derivativeB));
```
The derivatives need a reduction on the correct axis, we just have to remember that `AFderivative` has the same shape of `diff`, `(1,atomsInB,atomsInA,1)`.

#### GROUPA only
```c++
coordinationPlumed=0.5*coordination;
getToHost<calculateFloat>(0.5*AFvirial, DataInterface(virial));
getToHost<calculateFloat>(af::sum(AFderivative,2), DataInterface(derivativeA));

```
Here we need to remeber to not count twice everithing, since the common workflow considers each pair as if it belogns to two different groups.
So we just need to reduce le derivatives only along a single axis, and since we can choose, we reduce them along the axis that give us the possibility of skipping the sign inversion.



## Some caveat
At time of writing this the ArrayFire API (or at least, the following statement is based on how my compiler interprets the 3.9.0 API) has at least two overloads for `af::sum` that interest to us:
 - `AFAPI array 	sum (const array &in, const int dim=-1)` C++ Interface to sum array elements over a given dimension. 
 - `template<typename T > T sum (const array &in)` C++ Interface to sum array elements over all dimensions. 
 The problem is that the first one wiht its default argumens shadows the second one (this will be a "problem" when reducing the coordination)

## Limitations

`FIRECOORDINATION` and `FIRECOORDINATIONFLOAT` work more or less as the standard `COORDINATION`, except from:

 - work only with orthogonal pbcs or no pbcs at all
 - do not support the SWITCH keyword
   - and use the rational switch only with __even__ `NN` and `MM`

## Current TODO
 
 - Integrate the CI
