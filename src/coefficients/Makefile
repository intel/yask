SOURCES := fd_coeff_test.cpp fd_coeff.cpp
OBJS := $(SOURCES:.c=.o)
CXX := g++
CFLAGS := -std=c++11
INCLUDES := -I.
LDFLAGS :=
EXECUTABLE := fd_coeff_test

all: $(EXECUTABLE)

$(EXECUTABLE):$(OBJS)
	$(CXX) $(CFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDFLAGS) $(LIBS)

clean: 
	@rm -f $(EXECUTABLE) 
