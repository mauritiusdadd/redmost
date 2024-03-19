"""
Abstract apy to load different Qt backends

Note: Based on https://github.com/pytest-dev/pytest-qt
"""
import os
import sys
from typing import Dict, Union, Any
from types import ModuleType
from collections import namedtuple

try:
    import PyQt6
except (ModuleNotFoundError, ImportError):
    HAS_PYQT6 = False
else:
    HAS_PYQT6 = True

try:
    import PySide6
except (ModuleNotFoundError, ImportError):
    HAS_PYSIDE6 = False
else:
    HAS_PYSIDE6 = True

try:
    import PyQt5
except (ModuleNotFoundError, ImportError):
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


VersionTuple = namedtuple(
    typename="VersionTuple",
    field_names="qt_api, qt_api_version, runtime, compiled"
)

QT_APIS: Dict[str, str] = {
    x.lower(): x
    for x in ["PyQt6", "PySide6", "PyQt5", "PySide2"]
}


def _import(name):
    """Think call so we can mock it during testing"""
    return __import__(name)


def is_library_loaded(name):
    return name in sys.modules


def get_qt_api_from_env() -> Union[None, str]:
    api = os.environ.get("PYTEST_QT_API")
    if api is None:
        api = os.environ.get("QT_API")

    if api is not None:
        api = api.lower()
        if api not in QT_APIS.keys():  # pragma: no cover
            raise ValueError(
                f"Invalid value for $qt_api_name: {api}, "
                "expected one of {supported_apis}"
            )
    return api


def get_already_loaded_backend() -> Union[None, str]:
    for api, backend in QT_APIS.items():
        if is_library_loaded(backend):
            return api
    return None


class QtBackend:
    """
    Interface to the underlying Qt API currently configured.

    This object lazily loads all class references and other objects when
    the ``set_qt_api`` method gets called, providing a uniform way to
    access the Qt classes.
    """

    def __init__(self) -> None:
        self._import_errors = {}
        self.is_pyside: Union[None, bool] = None
        self.is_pyqt: Union[None, bool] = None
        self.qt_api_name: Union[None, str] = None

        self.QtCore = None
        self.QtGui = None
        self.QtTest = None
        self.QtWidgets = None
        self.QtCharts = None

        self.qInfo: Union[None, Any] = None
        self.qDebug: Union[None, Any] = None
        self.qWarning: Union[None, Any] = None
        self.qCritical: Union[None, Any] = None
        self.qFatal: Union[None, Any] = None

        self.Signal: Union[None, Any] = None
        self.Slot: Union[None, Any] = None
        self.Property: Union[None, Any] = None

        self.set_qt_api()

    def _guess_qt_api(self):  # pragma: no cover
        def _can_import(name):
            try:
                _import(name)
                return True
            except ModuleNotFoundError as e:
                self._import_errors[name] = str(e)
                return False

        # Note, not importing only the root namespace because
        # when uninstalling from conda, the namespace can still be there.
        for api, backend in QT_APIS.items():
            if _can_import(f"{backend}.QtCore"):
                return api
        return None

    def _import_module(self, module_name):
        _root_module = QT_APIS[self.qt_api_name]
        m = __import__(_root_module, globals(), locals(), [module_name], 0)
        return getattr(m, module_name)

    def set_qt_api(self, api=None):
        self.qt_api_name = (
            get_qt_api_from_env()
            or api
            or get_already_loaded_backend()
            or self._guess_qt_api()
        )

        self.is_pyside = self.qt_api_name in ["pyside2", "pyside6"]
        self.is_pyqt = self.qt_api_name in ["pyqt5", "pyqt6"]

        if not self.qt_api_name:  # pragma: no cover
            errors = "\n".join(
                f"  {module}: {reason}"
                for module, reason in sorted(self._import_errors.items())
            )
            raise ValueError(
                "Redmost requires either PySide2, PySide6, PyQt5 or PyQt6 "
                "installed.\n" + errors
            )

        self.QtCore = QtCore = self._import_module("QtCore")
        self.QtGui = self._import_module("QtGui")
        self.QtTest = self._import_module("QtTest")
        self.QtWidgets = self._import_module("QtWidgets")

        if self.qt_api_name == "pyqt5":
            self.QtCharts = self._import_module("QtChart")
        else:
            self.QtCharts = self._import_module("QtCharts")

        self._check_qt_api_version()

        # qInfo is not exposed in PySide2/6 (#232)
        if hasattr(QtCore, "QMessageLogger"):
            self.qInfo = lambda msg: QtCore.QMessageLogger().info(msg)
        elif hasattr(QtCore, "qInfo"):
            self.qInfo = QtCore.qInfo
        else:
            self.qInfo = None

        self.qDebug = QtCore.qDebug
        self.qWarning = QtCore.qWarning
        self.qCritical = QtCore.qCritical
        self.qFatal = QtCore.qFatal

        if self.is_pyside:
            self.Signal = QtCore.Signal
            self.Slot = QtCore.Slot
            self.Property = QtCore.Property
        elif self.is_pyqt:
            self.Signal = QtCore.pyqtSignal
            self.Slot = QtCore.pyqtSlot
            self.Property = QtCore.pyqtProperty
        else:
            assert False, "Expected either is_pyqt or is_pyside"

    def _check_qt_api_version(self):
        pass
        # Just a placeholder for now
        #
        # if self.is_pyside:
        #    return
        # elif self.QtCore.PYQT_VERSION == 0x060000:  # 6.0.0
        #    return

    @staticmethod
    def exec(obj, *args, **kwargs):
        # exec was a keyword in Python 2, so PySide2 (and also PySide6 6.0)
        # name the corresponding method "exec_" instead.
        #
        # The old _exec() alias is removed in PyQt6 and also deprecated as of
        # PySide 6.1:
        # https://codereview.qt-project.org/c/pyside/pyside-setup/+/342095
        if hasattr(obj, "exec"):
            return obj.exec(*args, **kwargs)
        return obj.exec_(*args, **kwargs)

    def get_versions(self):
        if self.qt_api_name == "pyside6":
            import PySide6

            version = PySide6.__version__

            return VersionTuple(
                "PySide6",
                version,
                self.QtCore.qVersion(),
                self.QtCore.__version__
            )
        elif self.qt_api_name == "pyside2":
            import PySide2

            version = PySide2.__version__

            return VersionTuple(
                "PySide2",
                version,
                self.QtCore.qVersion(),
                self.QtCore.__version__
            )
        elif self.qt_api_name == "pyqt6":
            return VersionTuple(
                "PyQt6",
                self.QtCore.PYQT_VERSION_STR,
                self.QtCore.qVersion(),
                self.QtCore.QT_VERSION_STR,
            )
        elif self.qt_api_name == "pyqt5":
            return VersionTuple(
                "PyQt5",
                self.QtCore.PYQT_VERSION_STR,
                self.QtCore.qVersion(),
                self.QtCore.QT_VERSION_STR,
            )

        raise ValueError(
            f"Internal error, unknown qt_api_name: {self.qt_api_name}"
        )

    def loadUiWidget(
        self,
        ui_filename: str,
        parent=None,
    ):
        """
        Load a UI file.

        :param ui_filename: Path of the UI file to load.
        :param parent: Parent widget. The default is None.
        :return ui: The widget loaded from the UI file.

        """

        ui_file_path: str = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'ui',
            ui_filename
        )

        if self.is_pyqt:
            uic = self._import_module("uic")
            ui = uic.loadUi(ui_file_path)
        elif self.is_pyside:
            uic = self._import_module("QtUiTools")
            loader = uic.QUiLoader()
            ui_file = self.QtCore.QFile(ui_file_path)
            ui_file.open(self.QtCore.QIODeviceBase.OpenModeFlag.ReadOnly)
            ui = loader.load(ui_file, parent)
            ui_file.close()
        else:
            raise ValueError("No valid GUI backend found!")

        return ui


qt_api = QtBackend()


def get_qapp():
    qapp = qt_api.QtWidgets.QApplication.instance()
    if qapp is None:
        # if it does not exist then a QApplication is created
        qapp = qt_api.QtWidgets.QApplication(sys.argv)
    return qapp
