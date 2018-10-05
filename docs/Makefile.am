if HAVE_DOXYGEN

directory = $(top_srcdir)/docs/man/man3/

man_MANS = $(directory)

$(directory):	UMUQ.doxyfile.stamp

doxyfile.stamp: UMUQ.Doxyfile	
	$(DOXYGEN) $^  
	cd latex && $(MAKE)	
	echo Timestamp > doxyfile.stamp

CLEANFILES = UMUQ.doxyfile.stamp

all-local:	UMUQ.doxyfile.stamp

clean-local:
	rm -rf $(top_srcdir)/docs/man

endif 
