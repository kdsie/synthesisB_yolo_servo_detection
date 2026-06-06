# Copyright (c) 2016 Adafruit Industries
# Author: Tony DiCola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from __future__ import division
import logging
import time
import math
import os
import fcntl
import glob


# Registers/etc:
PCA9685_ADDRESS    = 0x40
MODE1              = 0x00
MODE2              = 0x01
SUBADR1            = 0x02
SUBADR2            = 0x03
SUBADR3            = 0x04
PRESCALE           = 0xFE
LED0_ON_L          = 0x06
LED0_ON_H          = 0x07
LED0_OFF_L         = 0x08
LED0_OFF_H         = 0x09
ALL_LED_ON_L       = 0xFA
ALL_LED_ON_H       = 0xFB
ALL_LED_OFF_L      = 0xFC
ALL_LED_OFF_H      = 0xFD

# Bits:
RESTART            = 0x80
SLEEP              = 0x10
ALLCALL            = 0x01
INVRT              = 0x10
OUTDRV             = 0x04


logger = logging.getLogger(__name__)


class _LinuxI2CDevice(object):
    """Small /dev/i2c-* fallback used when Adafruit_GPIO is not installed."""

    I2C_SLAVE = 0x0703

    @staticmethod
    def _available_buses():
        buses = []
        for path in glob.glob("/dev/i2c-*"):
            try:
                buses.append(int(path.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                pass
        return sorted(buses)

    @classmethod
    def _open_bus(cls, address, busnum):
        fd = os.open(f"/dev/i2c-{busnum}", os.O_RDWR)
        try:
            fcntl.ioctl(fd, cls.I2C_SLAVE, address)
            return fd
        except Exception:
            os.close(fd)
            raise

    @classmethod
    def _probe_fd(cls, fd, busnum):
        # Probe the registers used during PCA9685 startup. Some Rockchip buses
        # accept the address ioctl but fail on real register access; those must
        # not be selected.
        for register in (MODE1, MODE2, PRESCALE, MODE1):
            os.write(fd, bytes([register & 0xFF]))
            data = os.read(fd, 1)
            if len(data) != 1:
                raise OSError(f"short read on /dev/i2c-{busnum}")

    @classmethod
    def _detect_bus(cls, address):
        errors = []
        for busnum in cls._available_buses():
            try:
                fd = cls._open_bus(address, busnum)
                try:
                    cls._probe_fd(fd, busnum)
                except OSError:
                    os.close(fd)
                    raise
                return busnum, fd
            except OSError as exc:
                errors.append(f"/dev/i2c-{busnum}@0x{address:02x}: {exc}")
        raise OSError("PCA9685 not found on /dev/i2c-*; " + "; ".join(errors))

    def __init__(self, address=None, busnum=None, **kwargs):
        address = kwargs.pop("address", address)
        busnum = kwargs.pop("busnum", busnum)

        if address is None:
            address_text = os.getenv("PCA9685_ADDRESS", "").strip()
            address = int(address_text, 0) if address_text else PCA9685_ADDRESS

        if busnum is None:
            bus_text = os.getenv("PCA9685_BUS", "").strip()
            busnum = int(bus_text, 0) if bus_text else None

        self.address = address
        if busnum is None:
            self.busnum, self._fd = self._detect_bus(self.address)
            print(f"PCA9685 detected on /dev/i2c-{self.busnum} address 0x{self.address:02x}")
        else:
            self.busnum = busnum
            self._fd = self._open_bus(self.address, self.busnum)
            self._probe_fd(self._fd, self.busnum)

    def write8(self, register, value):
        os.write(self._fd, bytes([register & 0xFF, value & 0xFF]))

    def readU8(self, register):
        os.write(self._fd, bytes([register & 0xFF]))
        return os.read(self._fd, 1)[0]

    def writeRaw8(self, value):
        os.write(self._fd, bytes([value & 0xFF]))

    def __del__(self):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None


def software_reset(i2c=None, **kwargs):
    """Sends a software reset (SWRST) command to all servo drivers on the bus."""
    if i2c is None:
        try:
            import Adafruit_GPIO.I2C as I2C
            i2c = I2C
        except ImportError:
            device = _LinuxI2CDevice(0x00, **kwargs)
            device.writeRaw8(0x06)
            return
    device = i2c.get_i2c_device(0x00, **kwargs)
    device.writeRaw8(0x06)  # SWRST


class PCA9685(object):
    """PCA9685 PWM LED/servo controller."""

    def __init__(self, address=None, i2c=None, **kwargs):
        """Initialize the PCA9685."""
        # Setup I2C interface for the device.
        if i2c is None:
            # Prefer the local /dev/i2c-* implementation on OrangePi. It avoids
            # Adafruit_GPIO choosing the wrong bus when multiple I2C buses exist.
            self._device = _LinuxI2CDevice(address, **kwargs)
        else:
            self._device = i2c.get_i2c_device(address, **kwargs)
        self.set_all_pwm(0, 0)
        self._device.write8(MODE2, OUTDRV)
        self._device.write8(MODE1, ALLCALL)
        time.sleep(0.005)  # wait for oscillator
        mode1 = self._device.readU8(MODE1)
        mode1 = mode1 & ~SLEEP  # wake up (reset sleep)
        self._device.write8(MODE1, mode1)
        time.sleep(0.005)  # wait for oscillator

    def set_pwm_freq(self, freq_hz):
        """Set the PWM frequency to the provided value in hertz."""
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        logger.debug('Setting PWM frequency to {0} Hz'.format(freq_hz))
        logger.debug('Estimated pre-scale: {0}'.format(prescaleval))
        prescale = int(math.floor(prescaleval + 0.5))
        logger.debug('Final pre-scale: {0}'.format(prescale))
        oldmode = self._device.readU8(MODE1);
        newmode = (oldmode & 0x7F) | 0x10    # sleep
        self._device.write8(MODE1, newmode)  # go to sleep
        self._device.write8(PRESCALE, prescale)
        self._device.write8(MODE1, oldmode)
        time.sleep(0.005)
        self._device.write8(MODE1, oldmode | 0x80)

    def set_pwm(self, channel, on, off):
        """Sets a single PWM channel."""
        self._device.write8(LED0_ON_L+4*channel, on & 0xFF)
        self._device.write8(LED0_ON_H+4*channel, on >> 8)
        self._device.write8(LED0_OFF_L+4*channel, off & 0xFF)
        self._device.write8(LED0_OFF_H+4*channel, off >> 8)

    def set_all_pwm(self, on, off):
        """Sets all PWM channels."""
        self._device.write8(ALL_LED_ON_L, on & 0xFF)
        self._device.write8(ALL_LED_ON_H, on >> 8)
        self._device.write8(ALL_LED_OFF_L, off & 0xFF)
        self._device.write8(ALL_LED_OFF_H, off >> 8)
