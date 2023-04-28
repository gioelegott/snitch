## Verification on CVA6

The `verification.py` script can be used to run a kernel in RTL simulation and verify it against a golden model in Python. 
The script starts the simulation as a subprocess and communicates with the front-end server of the simulation using FIFOS (named pipes).
After the simulation is complete, the Python script reads the kernel's result from the testbench memory of the simulation and compares it to the golden model.

### Verify a new kernel

In `golden_models.py`, add a golden model function for the kernel that returns the output matrix (or multiple matrices) and the name of the output variable as used in the C code of the kernel. 
The golden model function must have the same name as the kernel in the C code and takes the path to the kernel's header file as an argument.
The function `axpy()` in `golden_models.py` is an example of this.

Compile the hardware and software of the simulation and run the verification (when in `hw/system/occamy`):

```bash
python ../../ip/test/src/verification.py bin/occamy_top.vsim sw/build/<kernel_binary>
# For example to verify the axpy
python ../../ip/test/src/verification.py bin/occamy_top.vsim sw/build/axpy
```

![verification_3](https://user-images.githubusercontent.com/44872918/229832639-94022db5-a182-4962-997a-feeacbe0fa38.png)
