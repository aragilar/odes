import io
import os
from os.path import join
import sys
import textwrap

from distutils.log import info
from setuptools import Extension
from distutils.command.build_ext import build_ext as _build_ext

PKGCONFIG_CVODE = 'sundials-cvode-serial'
PKGCONFIG_IDA = 'sundials-ida-serial'
PKGCONFIG_CVODES = 'sundials-cvodes-serial'
PKGCONFIG_IDAS = 'sundials-idas-serial'


def write_pxi(filename, definitions):
    """
    Write a cython include file (.pxi), `filename`, with the definitions in the
    `definitions` mapping.
    """
    with io.open(filename, mode='w', encoding='utf-8') as pxi_file:
        for name, val in definitions.items():
            pxi_file.write(u"DEF {name} = {val}\n".format(name=name, val=val))
    return filename


def check_macro_true(cmd, symbol, headers=None, include_dirs=None):
    """
    Copied from numpy.distutils.command.config:config.check_macro_true, checks
    if macro is true
    """
    cmd._check_compiler()
    body = textwrap.dedent("""
        int main(void)
        {
        #if %s
        #else
        #error false or undefined macro
        #endif
            ;
            return 0;
        }""") % (symbol,)

    return cmd.try_compile(body, headers, include_dirs)


def check_macro_def(cmd, symbol, headers=None, include_dirs=None):
    """
    Based on numpy.distutils.command.config:config.check_macro_true, checks if
    macro is defined or not
    """
    cmd._check_compiler()
    body = textwrap.dedent("""
        int main(void)
        {
        #ifdef %s
        #else
        #error undefined macro
        #endif
            ;
            return 0;
        }""") % (symbol,)

    return cmd.try_compile(body, headers, include_dirs)


def get_sundials_config_pxi(include_dirs, dist):
    """
    Create pxi file containing some of sundials build config

    Don't ask why this is a function, something crazy about
    distutils/numpy not setting _setup_distribution at the right time or
    something...
    """
    SUNDIALS_CONFIG_H = "sundials/sundials_config.h"
    BASE_PATH = 'src/scikits_odes_sundials'

    config_cmd = dist.get_command_obj("config")

    # Get float type
    if check_macro_true(
        config_cmd,
        "SUNDIALS_DOUBLE_PRECISION", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_FLOAT_TYPE = '"double"'
        info("Found sundials built with double precision.")
    elif check_macro_true(
        config_cmd,
        "SUNDIALS_SINGLE_PRECISION", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_FLOAT_TYPE = '"single"'
        info("Found sundials built with single precision.")
    elif check_macro_true(
        config_cmd,
        "SUNDIALS_EXTENDED_PRECISION", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_FLOAT_TYPE = '"extended"'
        info("Found sundials built with extended precision.")
    else:
        # fall back to double
        SUNDIALS_FLOAT_TYPE = '"double"'
        info("Failed to find sundials precision, falling back to double...")

    # Get index (int) type
    if check_macro_true(
        config_cmd,
        "SUNDIALS_INT32_T", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_INDEX_SIZE = '"int32"'
        info("Found sundials built with int32.")
    elif check_macro_true(
        config_cmd,
        "SUNDIALS_INT64_T", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_INDEX_SIZE = '"64"'
        info("Found sundials built with int64.")
    else:
        # fall back to int64
        SUNDIALS_INDEX_SIZE = '"64"'
        info("Failed to find sundials index type, falling back to int64...")

    # Check for blas/lapack
    if check_macro_def(
        config_cmd,
        "SUNDIALS_BLAS_LAPACK_ENABLED", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        has_lapack = True
    else:
        has_lapack = False

    if check_macro_true(
        config_cmd,
        "SUNDIALS_MPI_ENABLED", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs,
    ):
        SUNDIALS_MPI_ENABLED = True
    else:
        SUNDIALS_MPI_ENABLED = False


    cfg = dict(
        float_type = SUNDIALS_FLOAT_TYPE,
        index_size = SUNDIALS_INDEX_SIZE,
        has_lapack = has_lapack,
        with_mpi = SUNDIALS_MPI_ENABLED,
    )

    return write_pxi(join(BASE_PATH, "sundials_config.pxi"), dict(
        SUNDIALS_FLOAT_TYPE=SUNDIALS_FLOAT_TYPE,
        SUNDIALS_INDEX_SIZE=SUNDIALS_INDEX_SIZE,
        SUNDIALS_BLAS_LAPACK=str(has_lapack),
    )), cfg


class build_ext(_build_ext):
    """
        Custom distutils command which encapsulates api_gen pre-building,
        Cython building, and C compilation.
    """
    def _get_cython_ext(self):
        import numpy
        base_path = 'src/scikits_odes_sundials'
        base_module = "scikits_odes_sundials"

        SUNDIALS_LIBRARIES = []
        CVODE_LIBRARIES = []
        IDA_LIBRARIES = []
        CVODES_LIBRARIES = []
        IDAS_LIBRARIES = []

        SUNDIALS_LIBRARY_DIRS = []
        CVODE_LIBRARY_DIRS = []
        IDA_LIBRARY_DIRS = []
        CVODES_LIBRARY_DIRS = []
        IDAS_LIBRARY_DIRS = []

        # numpy is required for all the extensions
        SUNDIALS_INCLUDE_DIRS = [numpy.get_include()]
        CVODE_INCLUDE_DIRS = []
        IDA_INCLUDE_DIRS = []
        CVODES_INCLUDE_DIRS = []
        IDAS_INCLUDE_DIRS = []

        SUNDIALS_LIBDIR = os.environ.get("SUNDIALS_LIBDIR")
        SUNDIALS_INCLUDEDIR = os.environ.get("SUNDIALS_INCLUDEDIR")
        SUNDIALS_INST_PREFIX = os.environ.get("SUNDIALS_INST")

        if SUNDIALS_LIBDIR or SUNDIALS_INCLUDEDIR:
            SUNDIALS_INCLUDE_DIRS.extend(
                [SUNDIALS_INCLUDEDIR] if SUNDIALS_INCLUDEDIR is not None else []
            )
            SUNDIALS_LIBRARY_DIRS.extend(
                [SUNDIALS_LIBDIR] if SUNDIALS_LIBDIR is not None else []
            )

        elif SUNDIALS_INST_PREFIX is not None:
            SUNDIALS_LIBRARY_DIRS.append(os.path.join(SUNDIALS_INST_PREFIX, "lib"))
            SUNDIALS_INCLUDE_DIRS.append(os.path.join(SUNDIALS_INST_PREFIX, "include"))
            info("SUNDIALS installation path set to `{}` via $SUNDIALS_INST.".format(
                SUNDIALS_INST_PREFIX))
        else:
            info("Searching for SUNDIALS path...")

            # use pkgconfig to find sundials
            try:
                import pkgconfig
                from pkgconfig.pkgconfig import PackageNotFoundError
                try:
                    cvode_pkgconf = pkgconfig.parse(PKGCONFIG_CVODE)
                    for d in cvode_pkgconf.get('library_dirs', []):
                        CVODE_LIBRARY_DIRS.append(str(d))
                    for d in cvode_pkgconf.get('include_dirs', []):
                        CVODE_INCLUDE_DIRS.append(str(d))
                    for lib in cvode_pkgconf.get('libraries', []):
                        CVODE_LIBRARIES.append(str(lib))

                    ida_pkgconf = pkgconfig.parse(PKGCONFIG_IDA)
                    for d in ida_pkgconf.get('library_dirs', []):
                        IDA_LIBRARY_DIRS.append(str(d))
                    for d in ida_pkgconf.get('include_dirs', []):
                        IDA_INCLUDE_DIRS.append(str(d))
                    for lib in ida_pkgconf.get('libraries', []):
                        IDA_LIBRARIES.append(str(lib))


                    cvodes_pkgconf = pkgconfig.parse(PKGCONFIG_CVODES)
                    for d in cvodes_pkgconf.get('library_dirs', []):
                        CVODES_LIBRARY_DIRS.append(str(d))
                    for d in cvodes_pkgconf.get('include_dirs', []):
                        CVODES_INCLUDE_DIRS.append(str(d))
                    for lib in cvodes_pkgconf.get('libraries', []):
                        CVODES_LIBRARIES.append(str(lib))

                    idas_pkgconf = pkgconfig.parse(PKGCONFIG_IDAS)
                    for d in idas_pkgconf.get('library_dirs', []):
                        IDAS_LIBRARY_DIRS.append(str(d))
                    for d in idas_pkgconf.get('include_dirs', []):
                        IDAS_INCLUDE_DIRS.append(str(d))
                    for lib in idas_pkgconf.get('libraries', []):
                        IDAS_LIBRARIES.append(str(lib))
                except (EnvironmentError, PackageNotFoundError) as e:
                    pass
            except ImportError:
                info("pkgconfig module not found, using preset paths")

        sundials_pxi, cfg = get_sundials_config_pxi(SUNDIALS_INCLUDE_DIRS,
                self.distribution)

        if not SUNDIALS_LIBRARIES:
            has_lapack = cfg['has_lapack']
            with_mpi = cfg["with_mpi"]
            if with_mpi:
                # use pkgconfig to find mpi
                try:
                    import pkgconfig
                    from pkgconfig.pkgconfig import PackageNotFoundError
                    try:
                        mpi_pkgconf = pkgconfig.parse("mpi")
                        for d in mpi_pkgconf.get('library_dirs', []):
                            SUNDIALS_LIBRARY_DIRS.append(str(d))
                        for d in mpi_pkgconf.get('include_dirs', []):
                            SUNDIALS_INCLUDE_DIRS.append(str(d))
                        for lib in mpi_pkgconf.get('libraries', []):
                            SUNDIALS_LIBRARIES.append(str(lib))
                    except (EnvironmentError, PackageNotFoundError) as e:
                        pass
                except ImportError:
                    info("pkgconfig module not found, using preset paths")
                    
            # SUNDIALS core types/context, needed starting with 7.0
            SUNDIALS_LIBRARIES.append('sundials_core') 

            # This is where to put N_vector codes (currently only serial is
            # supported)
            SUNDIALS_LIBRARIES.append('sundials_nvecserial')
            # SUNDIALS_LIBRARIES.append('sundials_nvecopenmp')
            # SUNDIALS_LIBRARIES.append('sundials_nvecparallel')
            # SUNDIALS_LIBRARIES.append('sundials_nvecparhyp')
            # SUNDIALS_LIBRARIES.append('sundials_nvecpetsc')
            # SUNDIALS_LIBRARIES.append('sundials_nvecpthreads')

            # This is where to put SUNLinearSolver codes (klu not supported
            # yet)
            if has_lapack:
                SUNDIALS_LIBRARIES.append('sundials_sunlinsollapackband')
                SUNDIALS_LIBRARIES.append('sundials_sunlinsollapackdense')

            SUNDIALS_LIBRARIES.append('sundials_sunlinsolband')
            SUNDIALS_LIBRARIES.append('sundials_sunlinsoldense')
            SUNDIALS_LIBRARIES.append('sundials_sunlinsolpcg')
            SUNDIALS_LIBRARIES.append('sundials_sunlinsolspbcgs')
            SUNDIALS_LIBRARIES.append('sundials_sunlinsolspfgmr')
            SUNDIALS_LIBRARIES.append('sundials_sunlinsolspgmr')
            SUNDIALS_LIBRARIES.append('sundials_sunlinsolsptfqmr')
            # SUNDIALS_LIBRARIES.append('sundials_sunlinsolklu')

            # This is where to put SUNMatrix codes
            SUNDIALS_LIBRARIES.append('sundials_sunmatrixband')
            SUNDIALS_LIBRARIES.append('sundials_sunmatrixdense')
            SUNDIALS_LIBRARIES.append('sundials_sunmatrixsparse')

        if not IDA_LIBRARIES:
            IDA_LIBRARIES.append('sundials_ida')

        if not CVODE_LIBRARIES:
            CVODE_LIBRARIES.append('sundials_cvode')

        if not IDAS_LIBRARIES:
            IDAS_LIBRARIES.append('sundials_idas')

        if not CVODES_LIBRARIES:
            CVODES_LIBRARIES.append('sundials_cvodes')

        CVODE_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        IDA_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        CVODES_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        IDAS_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        CVODE_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        IDA_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        CVODES_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        IDAS_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        CVODE_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)
        IDA_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)
        CVODES_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)
        IDAS_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)

        return [
            Extension(
                base_module + '.' + "common_defs",
                sources = [join(base_path, 'common_defs.pyx')],
                include_dirs=SUNDIALS_INCLUDE_DIRS,
                library_dirs=SUNDIALS_LIBRARY_DIRS,
                libraries=SUNDIALS_LIBRARIES,
            ),
            Extension(
                base_module + '.' + "cvode",
                sources = [join(base_path, 'cvode.pyx')],
                include_dirs=CVODE_INCLUDE_DIRS,
                library_dirs=CVODE_LIBRARY_DIRS,
                libraries=CVODE_LIBRARIES,
            ),
            Extension(
                base_module + '.' + "ida",
                sources = [join(base_path, 'ida.pyx')],
                include_dirs=IDA_INCLUDE_DIRS,
                library_dirs=IDA_LIBRARY_DIRS,
                libraries=IDA_LIBRARIES,
            ),
            Extension(
                base_module + '.' + "cvodes",
                sources = [join(base_path, 'cvodes.pyx')],
                include_dirs=CVODES_INCLUDE_DIRS,
                library_dirs=CVODES_LIBRARY_DIRS,
                libraries=CVODES_LIBRARIES,
            ),
            Extension(
                base_module + '.' + "idas",
                sources = [join(base_path, 'idas.pyx')],
                include_dirs=IDAS_INCLUDE_DIRS,
                library_dirs=IDAS_LIBRARY_DIRS,
                libraries=IDAS_LIBRARIES,
            ),
        ]


    def run(self):
        """ Distutils calls this method to run the command """
        from Cython.Build import cythonize
        self.extensions = cythonize(
            self._get_cython_ext(),
            compiler_directives={
                'language_level': 3,
            },
        )
        _build_ext.run(self) # actually do the build

