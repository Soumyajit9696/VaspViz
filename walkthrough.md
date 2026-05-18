# Equation of State (EOS) Fitter Implementation

The Equation of State (EOS) Fitter has been successfully implemented and integrated into VaspViz as a brand new isolated module (`eos_fitter.py`). No other working portions of the code were modified!

## What was Accomplished

1. **New UI Tab**: Added an "EOS Fitter" tab (Index 10) to the main `QTabWidget` and a corresponding button in the main sidebar.
2. **Data Input System**:
   - **Load from Files**: A button allows you to select multiple `vasprun.xml` files. The module automatically parses the final lattice vectors to compute the cell volume ($V$) and extracts the total energy ($E$) for each file.
   - **Manual Paste**: A `QPlainTextEdit` area allows you to instantly copy-paste raw $V$ vs $E$ columns from Excel, `OUTCAR` grep outputs, or any other source.
3. **Robust Model Fitting**:
   - Implemented standard models using `scipy.optimize.curve_fit`.
   - Supports: **Birch-Murnaghan (3rd Order)**, **Murnaghan**, **Rose-Vinet**, and a basic **Parabola**.
   - Includes fallback logic to automatically use a parabolic pre-fit to generate intelligent initial guesses for the more complex non-linear EOS equations, preventing `Optimal parameters not found` errors.
4. **Data Visualization**:
   - A Matplotlib canvas displays your raw DFT data points.
   - The fitted EOS curve is plotted as a smooth line.
   - A marker denotes the exact computed minimum.
5. **Results Extraction**:
   - Dynamically calculates and displays:
     - **$V_0$**: Equilibrium Volume (Å³)
     - **$E_0$**: Minimum Energy (eV)
     - **$B_0$**: Bulk Modulus (GPa)
     - **$B'_0$**: Pressure derivative of the bulk modulus.

## How to Test
1. Restart VaspViz.
2. Open the new "EOS Fitter" tab via the bottom of the left-hand menu.
3. Either click "Load vasprun.xml files" and select a few structural optimization steps, or type/paste some fake dummy data like:
```text
30.0   -120.1
31.0   -121.5
32.0   -121.8
33.0   -121.2
34.0   -119.8
```
4. The tool will instantly fit the curve and output the equilibrium volume and Bulk Modulus!
