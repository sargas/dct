from __future__ import division
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, run_module_suite, dec, assert_approx_equal, assert_allclose
from dct import lmi
import numpy.ma.testutils as matu
from contextlib import contextmanager
from astropy.io import fits
from itertools import product

def test_offset():
    X,Y=np.mgrid[-5:5:0.05,-5:5:0.05]
    Z=np.sqrt(X**2+Y**2)+np.sin(X**2+Y**2)
    Z2 = Z.copy()
    for i in range(15):
        dx, dy = lmi.find_offset(Z, Z2)
        dx2, dy2 = lmi.find_offset(Z2, Z)
        assert_array_equal([dx, dx2], [0,0])
        assert_array_equal([dy, dy2], [i, -i])
        Z2 = np.pad(Z2, ((0,0),(1,0)), mode='constant')[:, :-1]

    Z2 = Z.copy()
    for i in range(15):
        dx, dy = lmi.find_offset(Z, Z2)
        dx2, dy2 = lmi.find_offset(Z2, Z)
        assert_array_equal([dx, dx2], [i,-i])
        assert_array_equal([dy, dy2], [0,0])
        Z2 = np.pad(Z2, ((1,0),(0,0)), mode='constant')[:-1, :]

    Z2 = Z.copy()
    for i in range(15):
        dx, dy = lmi.find_offset(Z, Z2)
        dx2, dy2 = lmi.find_offset(Z2, Z)
        assert_array_equal([dx, dx2], [i,-i])
        assert_array_equal([dy, dy2], [i, -i])
        Z2 = np.pad(Z2, ((1,0),(1,0)), mode='constant')[:-1, :-1]

def test_combine_images_same_size_clipped():
    Z = np.array([[1,2,3,4],[2,3,4,5],[2,3,4,5]])
    Z2 = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    tests = [
             [0, 0, Z+Z2],
             [2, 0,  np.array( [[4,5],[6,7],[7,8]])],
             [-2, 0, np.array([[2, 3],[4, 5],[5,6]])],
             [0, 2, np.array([[3, 4, 5, 6]])],
             [0, -2, np.array([[4, 5, 6, 7]])],
             [2, 2, np.array([[5,6]])]
            ]

    for test in tests:
        c = lmi._combine_images(Z, Z2, [test[0], test[1]], clip=True)
        assert_equal(c, test[2])

def test_combine_images_same_size_not_clipped():
    Z = np.array([[1,2,3,4],[2,3,4,5],[2,3,4,5]])
    Z2 = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    tests = [
             [0, 0, Z+Z2],
             [2, 0,  np.array([[0,0,4,5],[0,0,6,7],[0,0,7,8]])],
             [-2, 0, np.array([[2,3,0,0],[4,5,0,0],[5,6,0,0]])],
             [0, 2, np.array([[0,0,0,0],[0,0,0,0],[3,4,5,6]])],
             [0, -2, np.array([[4, 5, 6, 7],[0,0,0,0],[0,0,0,0]])],
             [2, 2, np.array([[0,0,0,0],[0,0,0,0],[0,0,5,6]])]
            ]

    for test in tests:
        c = lmi._combine_images(Z, Z2, [test[0], test[1]], clip=False)
        assert_equal(c, test[2])

def test_zero_pad_to_same_size():
    Z = np.array([[1,2,3,4],[2,3,4,5],[2,3,4,5]])
    Z2 = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    tests = [
             [Z, Z2, Z, 0, 0],
             [Z[:2], Z2, np.array([[0,0,0,0],[1,2,3,4],[2,3,4,5]]), 0, -1],
             [Z[:,:3], Z2, np.array([[0,1,2,3],[0,2,3,4],[0,2,3,4]]), -1, 0],
             [Z[1:,1:], Z2, np.array([[0,0,0,0],[0,3,4,5],[0,3,4,5]]), -1, -1]
            ]

    for test in tests:
        c = lmi._zero_pad_to_same_size(test[0], test[1])
        c2 = lmi._zero_pad_to_same_size(test[1], test[0])
        assert_equal(c, (test[2], test[1], [test[3], test[4]]))
        assert_equal(c2, (test[1], test[2], [-test[3], -test[4]]))

@contextmanager
def create_fake_fits_reader(hdu):
    old_methods = {"getdata": lmi.fits.getdata, "getheader": lmi.fits.getheader }
    try:
        lmi.fits.getdata = lambda x : hdu.data
        lmi.fits.getheader = lambda x : hdu.header
        yield
    finally:
        lmi.fits.getdata = old_methods['getdata']
        lmi.fits.getheader = old_methods['getheader']

def test_combine_flat():
    with create_fake_fits_reader(fits.PrimaryHDU(5*np.ones( (5,5) ))):
        flat = lmi.combine_flat([None])
    assert_allclose(flat.data, np.ones(flat.shape))

    with create_fake_fits_reader(fits.PrimaryHDU( np.array([[1,1,1],[2,2,2],[3,3,3]]) )):
        flat = lmi.combine_flat([None])
    assert_allclose(flat.data, [[0.5, 0.5, 0.5],[1,1,1],[1.5,1.5,1.5]])

def test_combine_flat_is_normalized():
    old_random_state = np.random.get_state()
    try:
        for i in range(50):
            np.random.seed(i)
            flat_data = np.random.randint(np.iinfo('i2').max, size=(50,50))
            with create_fake_fits_reader(fits.PrimaryHDU(flat_data)):
                flat = lmi.combine_flat([None])
            assert_approx_equal(flat.data.mean(), 1)
            assert_allclose(flat.data * flat_data.mean(), flat_data)
    finally:
        np.random.set_state(old_random_state)

def test_subtract_bias():
    bias = fits.PrimaryHDU(5 * np.ones( (5,5) ) )
    with create_fake_fits_reader(fits.PrimaryHDU(6.5*np.ones( (5,5) ))):
        img = lmi.open_image([None], bias=bias, medium_subtract=False)
    assert_allclose(img.data, 1.5*np.ones(img.shape))

def test_combine_flat_with_bias():
    bias = fits.PrimaryHDU(0.2 * np.ones( (5,5) ) )
    with create_fake_fits_reader(fits.PrimaryHDU(5*np.ones( (5,5) ))):
        flat = lmi.combine_flat([None], bias)
    assert_allclose(flat.data, np.ones(flat.data.shape))

def test_divide_flat():
    flat = fits.PrimaryHDU([[0.5,0.5,0.5],[1.5,1.5,1.5]])
    with create_fake_fits_reader(fits.PrimaryHDU([[5,5,5],[10,10,10]])):
        img = lmi.open_image([None], flat=flat, medium_subtract=False)
    assert_allclose(img.data, [[10,10,10],[20/3, 20/3, 20/3]])

def test_divide_flat_and_subtract_bias():
    flat = fits.PrimaryHDU(np.array([[1,1,1],[2,2,2],[3,3,3]])/2.0)
    bias = fits.PrimaryHDU(2 * np.ones( (3,3) ) )
    with create_fake_fits_reader(fits.PrimaryHDU([[5,5,5],[10,10,10],[15,15,15]])):
        img = lmi.open_image([None], flat=flat, bias=bias, medium_subtract=False)
    assert_allclose(img.data, [[6, 6, 6], [8, 8, 8], [26/3, 26/3, 26/3]])

def test_divide_flat_and_subtract_bias_random():
    a = np.outer(np.arange(40)+1, np.linspace(-2, 2, 50)) + 100
    flat = fits.PrimaryHDU(a / a.mean())
    b = np.ones( a.shape )
    for i in range(len(b)): b[i] = i
    bias = fits.PrimaryHDU(b)

    old_random_state = np.random.get_state()
    try:
        for seed in range(50):
            np.random.seed(seed)
            c = np.random.randint(b.max()+10, np.iinfo('i2').max, size=b.shape)

            with create_fake_fits_reader(fits.PrimaryHDU(c)):
                img = lmi.open_image([None], flat=flat, bias=bias, medium_subtract=False)
            assert_allclose(img.data, (c - b)/flat.data)
    finally:
        np.random.set_state(old_random_state)

def test_median_subtract():
    a = fits.PrimaryHDU(2 * np.ones( (3,3) ) ) # median == 2
    with create_fake_fits_reader(a):
        img = lmi.open_image([None], medium_subtract=True)
        assert_allclose(img.data, np.zeros(a.data.shape))

    b = fits.PrimaryHDU([[1,1,1],[0,0,0],[-1,-1,-1]]) # median = 0
    with create_fake_fits_reader(b):
        img = lmi.open_image([None], medium_subtract=True)
        assert_allclose(img.data, b.data)

if __name__ == "__main__":
    run_module_suite()
