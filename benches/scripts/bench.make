
test:
	@env PLUMED_MAKE=$(MAKE) ../../scripts/run

clean:
	rm -fr tmp/ report.txt

valgrind:
	../../scripts/run -v

testclean:
	$(MAKE) test
	rm -fr tmp/

