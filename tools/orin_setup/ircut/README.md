# Header pin 7 as a GPIO output (IR-cut filter signal)

On JetPack 6, `gpioset` on a 40-pin header pin returns success but the pin never changes: the pinmux is set at
boot and the kernel cannot switch the pin to output at run time. `pin7-gpio.dts` sets pin 7 (`soc_gpio59_pac6`,
gpiochip0 line 144) to GPIO with the same properties jetson-io writes.

The Orin loads its kernel, device tree and overlays from the SD card (`/dev/mmcblk0p1`); the system runs from the
NVMe. Install on both `/boot`s so they stay the same. The SD card's `extlinux.conf` boots the `JetsonIO` entry,
which already loads the IMX219 camera overlay; the NVMe copy has the same `JetsonIO` entry (its DEFAULT is
`primary`).

Build and install on both (needs sudo), then reboot:

```sh
cd ~/Projects/Party-Pose/tools/orin_setup/ircut
dtc -@ -I dts -O dtb -o pin7-gpio.dtbo pin7-gpio.dts
for B in /media/thesa/752d27c0-5370-43a2-9666-6f6c7e937dce/boot /boot; do
  sudo cp $B/extlinux/extlinux.conf $B/extlinux/extlinux.conf.before-pin7
  sudo cp pin7-gpio.dtbo $B/pin7-gpio.dtbo
  sudo sed -i 's#^\(\s*OVERLAYS /boot/tegra234-p3767-camera-p3768-imx219-dual.dtbo\)$#\1,/boot/pin7-gpio.dtbo#' $B/extlinux/extlinux.conf
  grep OVERLAYS $B/extlinux/extlinux.conf
done
sudo reboot
```

To undo: on both, copy `extlinux.conf.before-pin7` back over `extlinux.conf` and reboot.

Check after the reboot: `gpioset --mode=time --sec=5 gpiochip0 144=1` should put about 3.3 V on pin 7.
