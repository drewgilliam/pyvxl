cmake_minimum_required(VERSION 3.5.1)
project("pyvxl-contrib-brad")

# Add pybind11 module
pybind11_add_module(pybrad pybrad.h pybrad.cxx)

# Link to vxl library
target_link_libraries(pybrad PRIVATE brad)

# Set names
set_target_properties(pybrad PROPERTIES OUTPUT_NAME "_brad")

# install the .so file to the python install dir
install(TARGETS pybrad DESTINATION ${PYTHON_SITE}/vxl/contrib/brad)

# auto generate __init__ file
install(CODE "file(WRITE ${PYTHON_SITE}/vxl/contrib/brad/__init__.py \"from ._brad import *\")")

