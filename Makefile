CXX = g++
# Safety and debugging flags:
# -Wall: Enable all warnings
# -Wextra: Enable additional warnings
# -g: Generate debug info
# -fsanitize=address: Enable Address Sanitizer to detect memory errors
# -fstack-protector-all: Enable stack protection
# -D_FORTIFY_SOURCE=2: Use fortify source for additional runtime checks
# -Iinclude and -Iexternal: Include directories
CXXFLAGS = -Wall -Wextra -g -fsanitize=address -fstack-protector-all \
      -D_FORTIFY_SOURCE=2 -Iinclude -Iexternal -std=c++17 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = main

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
