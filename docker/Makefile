
.PHONY: ubuntu plumed2.tgz clean fedora38 rocky8

ubuntu: plumed2.tgz
	docker build -t plumed --progress=plain .

fedora38: plumed2.tgz
	docker build -t plumed --progress=plain -f fedora38 .

rocky8: plumed2.tgz
	docker build -t plumed --progress=plain -f rocky8 .

plumed2.tgz:
	cd ../ ; git archive HEAD -o "$(CURDIR)"/plumed2.tgz --prefix plumed2/ ; cd -

clean:
	rm -f plumed2.tgz
