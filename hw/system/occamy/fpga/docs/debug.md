__Using both JTAG chains at the same time :__
If you use the HS2 JTAG dongle, Vivado may block the access to it (for OpenOCD) because it automatically scan every JTAG chain. To avoid this, open an instance of `hw_server` with a few parameters in an init file :
```
set auto-open-ports 0
set always-open-jtag 0
set jtag-port-filter Xilinx
```

__Debugging the bootrom :__
The bootrom is wrtitten is a .coe file at `occamy/fpga/bootrom/bootrom.spl`, it is a concatenation of the primary bootloader with the device tree, and the secondary bootloader from u-boot.
When changing any of these elements to debug (for instance the device tree), the bootrom content requires to re-synthetize and implement the design before being updated.

A solution to debug the bootrom faster is to use the scratchpad memory in `0x7000000` (see Occamy's memory map).

First change the bootrom addresses to `0x70000000` :
```
riscv64-unknown-elf-objcopy --change-addresses=0x6f000000 bootrom.elf bootrom_scratch.elf
riscv64-unknown-elf-objdump -d bootrom_scratch.elf > bootrom_scratch.dump
```
Note, if your previous bootrom is still loaded in `0x1000000` then the u-boot SPL will still execute after reset. The best thing is to let the bootloader run and to stop it at the console message `Hit any key to stop autoboot`. Then you can start openOCD and GDB :
```
riscv64-unknown-elf-gdb
target extended-remote :3333
file bootrom_scratch.elf (in case of error try restarting GDB two or three times)
load
# To start back from bootrom
set $pc=0x70000000
set $priv=3
```

Now you can easily change your device tree, recompile `bootrom_scratch.elf` reload it into scratch, and set the PC to it.

__Booting Linux as a payload :__
A direct way to boot linux is to embbed it directly in the OpenSBI binary. See `osbi_linux` target in the Ariane SDK Makefile.

__Debugging the kernel or busybox :__
A good way to debug the kernel ( `buildroot/output/build/linux-ariane-v0.7` ) or Busybox ( `buildroot/output/build/busybox-1.33.0` ) is to include directly breakpoints in the code ( with the asm ).

__Write a simple "hello world" init process :__
If something does not work in Busybox init, you can create a simple init program containng a `asm volatile (ebreak)` and a `printf` as described in the [Busybox FAQ](https://busybox.net/FAQ.html#init). To do so, write your C code and compile it with
```
$(RISCV)/bin/riscv64-unknown-linux-gnu-gcc -Og -g -static my_init.c -o buildroot/output/target/bin/my_init
```
Then change `CONFIG_CMDLINE="rdinit=/bin/my_init init=/bin/my_init"` in your `linux_defconfig` and your program will be executed at the end of the boot sequence.

__Python version :__
When using `occamygen.py` :
```
Traceback (most recent call last):
  ...
  File "/usr/lib64/python3.6/subprocess.py", line 423, in run
    with Popen(*popenargs, **kwargs) as process:
TypeError: __init__() got an unexpected keyword argument 'text'
make: *** [update-source] Error 1
```
Update your python version (using miniconda for instance).

__Bender warnings :__
When updating the sources :
```
...
warning: Name issue with "common_cells", `export_include_dirs` not handled
	Could relate to name mismatch, see `bender update`
...
```
Remove `Bender.lock` and `.bender`