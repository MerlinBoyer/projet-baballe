CPPFLAGS=-I/opt/opencv/include 
CXXFLAGS=-Wall -Wextra 
LDFLAGS=-Wl,-R/opt/opencv/lib -L/opt/opencv/lib `pkg-config --libs opencv`
LDLIBS=\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_highgui
BIN=\
	find
	

.PHONY: all 
all: $(BIN)

.PHONY: merlin
merlin: find_a_ball\
	test_method_merlin

.PHONY: yann
yann: hsv

.PHONY: clean
clean:
	$(RM) *~ *.png

.PHONY: cleanall
cleanall: clean
	$(RM) $(BIN) *.o *.pdf
