# KLT Algorithm - Organized Structure

This is the Kanade-Lucas-Tomasi (KLT) feature tracking algorithm, now organized into a proper folder structure.

## Directory Structure

```
klt/
├── src/                    # Source code files
│   ├── *.c                # C source files
│   └── main.cpp           # C++ main file
├── include/                # Header files
│   ├── *.h                # Header files
├── input/                  # Input images
│   ├── img*.pgm           # Input PGM images
├── output/                 # Output files
│   ├── feat*.ppm          # Feature visualization images
│   ├── features.*         # Feature data files
│   └── example3           # Compiled executable
├── build/                  # Build system
│   ├── Makefile           # Unix/Linux makefile
│   ├── build.bat          # Windows build script
│   ├── libklt.a           # Static library
│   └── *.vcproj, *.sln    # Visual Studio project files
├── doc/                    # Documentation
│   └── *.html             # HTML documentation
└── matlab_interface/       # MATLAB interface
    └── *.m                # MATLAB scripts
```

## Building the Project

### Windows
```bash
cd build
build.bat
```

### Unix/Linux
```bash
cd build
make
```

## Running Example3

The main example (example3.c) demonstrates:
- Finding 150 best features in an image
- Tracking them through the next 9 images
- Saving feature visualizations as PPM files
- Saving feature data as text files

To run:
```bash
cd build
./example3        # Unix/Linux
example3.exe      # Windows
```

## File Organization Details

- **Source files** (`src/`): All `.c` files moved here with updated include paths
- **Header files** (`include/`): All `.h` files moved here
- **Input images** (`input/`): All `img*.pgm` files moved here
- **Output files** (`output/`): All generated files (feat*.ppm, features.*) go here
- **Build system** (`build/`): Makefile and build scripts with updated paths

## Key Changes Made

1. **Include paths**: Updated all `#include` statements to use `../include/` prefix
2. **File paths**: Updated example3.c to read from `../input/` and write to `../output/`
3. **Makefile**: Updated to work with new directory structure
4. **Build script**: Created Windows batch file for easy compilation

## Dependencies

- GCC compiler (or compatible C compiler)
- Standard C library
- Math library (-lm)

The project should now build and run without dependency issues in the new organized structure.
