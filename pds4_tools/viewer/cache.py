from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import shutil

from ..utils.logging import logger_init

from ..extern import appdirs

# Initialize the logger
logger = logger_init()

#################################


def init_cache():
    """
    Initializes necessary settings for PDS4 Viewer cache.
    These should be initialized as early as possible into the Viewer startup sequence.

    Returns
    -------
    None
    """

    # Do nothing unless the application has been frozen (e.g. via PyInstaller)
    if not hasattr(sys, 'frozen'):
        return

    # Set MPL cache
    _set_mpl_cache()

    # Write current version out to cache file
    # (must be done after any checks that depend on knowing last cached version, such as setting MPL cache)
    _write_cache_version()


def _get_cache_dir():
    """ Obtain location to store cache files for frozen version of the Viewer.

    By default, this uses use appdirs to resolve the user's application directory for all OS'. E.g.::

       Windows this usually C:/Users/<username>/AppData/local/<appname>
       Mac this is usually ~/Library/Application Support/<appname>
       Linux this is usually ~/.local/share/<AppName> or XDG defined

    A directory for the viewer cache is made inside the user's application directory.

    The environment variable ``PDS4VIEWERCACHEDIR`` may be used to specify an alternate directory.

    Returns
    -------
    str or unicode
        Path to directory used to store cache files for frozen version of the Viewer.
    """

    environ_cache_dir = os.environ.get('PDS4VIEWERCACHEDIR')

    if environ_cache_dir:
        cache_dir = environ_cache_dir

    else:
        cache_dir = appdirs.user_data_dir(appname=str('pds4_viewer'), appauthor=False)

    return cache_dir


def _get_current_version():
    """
    Returns
    -------
    str or unicode
        String containing the current version the Viewer.
    """

    return sys.modules['pds4_tools'].__version__


def _get_cache_version():
    """
    Returns
    -------
    str, unicode or None
        String containing the version of the Viewer used the last time its frozen version was run.
        None will result if this is the first time a frozen version is being run, or if the file storing
        the last version is inaccessible.
    """

    cache_dir = _get_cache_dir()
    version_file = os.path.join(cache_dir, 'version')
    try:

        with open(version_file, 'r') as file_handler:
            cached_version_string = file_handler.read()

    except (OSError, IOError) as e:

        logger.warning('Unable to read version from cache file: {0}'.format(str(e)))
        cached_version_string = None

    return cached_version_string


def _write_cache_version():
    """
    Writes a string containing the current version the Viewer to a cache file.

    Returns
    -------
    bool
        True if successfully written to file, False if an exception occurred.
    """

    cache_dir = _get_cache_dir()
    version_file = os.path.join(cache_dir, 'version')
    current_version = _get_current_version()

    try:

        with open(version_file, 'w') as file_handler:
            file_handler.write(current_version)

    except (OSError, IOError) as e:
        logger.warning('Unable to write version to cache file: {0}'.format(str(e)))
        return False

    return True


def _set_mpl_cache():
    """
    Matplotlib has a cache directory. When PDS4 Viewer code is frozen, e.g. via PyInstaller, it would be
    good if this was the same directory each time the application was opened because there is a fair delay
    in creating this cache. This method takes care of ensuring this is the case.

    Notes
    -----
    MPL tries to create aspects of this cache as soon as it believes it's needed, which is often
    as soon as some MPL import is called. Therefore it is wise to set this cache directory early.

    Setting the MPL cache when not frozen is not necessary because MPL will find a reliable place on
    its own. However, when the application is frozen, it is unpacked into a temporary directory on start
    up, and MPL is generally unable to find a reliable place in this circumstance, instead extracting
    inside the temporary directory it's started from. Since this temporary directory may well change unless
    it is fixed, MPL will try to create a new cache directory in a different place.

    Amongst other things, MPL will store a font cache, which includes the physical location of fonts.
    MPL also ships with the fonts it uses by default. Upon trying to locate the necessary fonts, it will
    often find the fonts it ships with. When frozen, these fonts will be located in the temporary directory
    it is extracted to. Therefore, although we may create a permanent place for the cache, it will not be
    correct if the correct font locations change each time the application is run (MPL will automatically
    rebuild its cache if a font location does not exist). To fix this, we also copy matplotlib's fonts to
    the cache directory and instruct it to look there.

    Returns
    -------
    None
    """

    # Obtain the cache directory for the Viewer
    app_cache_dir = _get_cache_dir()

    # Obtain the cache directory for MPL (inside the Viewer's cache directory)
    mpl_cache_dir = os.path.join(app_cache_dir, 'mpl-data')

    # Obtain the frozen location of 'mpl-data'
    mpl_frozen_dir = os.path.join(sys._MEIPASS, 'mpl-data')

    # Copy mpl-data to cache if necessary and possible. See notes docstring above for why this is necessary.
    # To save time, we copy only if Viewer version has changed since last time mpl-data was copied
    error_occurred = False
    if _get_cache_version() != _get_current_version():

        logger.warning('Copying mpl-data to enable permanent MPL cache. This should only run once, '
                       'on the initial run of each new version of PDS4 Viewer.')

        try:

            if os.path.exists(mpl_cache_dir):
                shutil.rmtree(mpl_cache_dir, ignore_errors=True)

            shutil.copytree(mpl_frozen_dir, mpl_cache_dir)

        except (OSError, IOError, shutil.Error) as e:
            error_occurred = True
            logger.warning('Unable to set MPL datapath. MPL cache may not work. '
                           'Received: {0}'.format(str(e)))

    if not error_occurred:

        # Set environment variable used by MPL to look for config and cache to the above defined cache folder
        os.environ['MPLCONFIGDIR'] = mpl_cache_dir

        # Import matplotlib now that cache dir has been set and mpl-data copied, and set datapath
        import matplotlib as mpl
        mpl.rcParams['datapath'] = mpl_cache_dir


# def get_recently_opened():
#
#
# def write_recently_opened(path_to_file):
#
#
