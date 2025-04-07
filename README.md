# PyGlyph: 3D ASCII Model Renderer

<p align="center">
  <img src="PyGlyph.png" alt="drawing" width="300"/>
</p>

PyGlyph is a Python-based 3D ASCII renderer capable of visualizing `.obj` models using ASCII art. Built using Pygame, Numpy, and Numba, PyGlyph offers smooth rendering and shading of complex 3D models.

![Date Uploaded](https://img.shields.io/badge/Date%20Uploaded-April%206,%202025-orange)
![Downloads](https://img.shields.io/badge/Downloads-0-blue)

## Features :rocket:

- **OBJ File Loading**: Supports standard `.obj` file formats.
- **ASCII Rendering**: Visualizes models using customizable ASCII character sets.
- **Smooth Shading**: Calculates vertex normals for realistic lighting.
- **Real-Time Rotation**: Supports continuous rotation along X and Y axes.
- **Ambient Lighting**: Adds subtle ambient lighting for depth perception.
- **Parallel Processing**: Uses Numba for optimized rendering performance.

## Requirements :construction:

- `pygame`
- `numpy`
- `numba`

Install dependencies using:

```bash
pip install pygame numpy numba
```

## Usage :gear:

:warning: place `model.obj` file in the same folder. If you need to convert your model from `.stl` you can use this website
<img src="Convert 3D.png" alt="drawing" width="100" />

```bash
python pyglyph.py
```

## Acknowledgments :sparkles:

This project was inspired by **AIKON's 3D donut ASCII renderer** written in C.  
You can check out AIKON's original work [here](https://www.a1k0n.net/2011/07/20/donut-math.html).
