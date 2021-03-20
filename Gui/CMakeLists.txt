project(Gui LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(
    SOURCES
    docks/ui/CameraDock.ui
    docks/ui/OutputDock.ui
    docks/ui/PPMDock.ui
    docks/ui/RenderInformationDock.ui
    docks/ui/SceneDock.ui
    docks/CameraDock.cpp
    docks/OutputDock.cpp
    docks/PPMDock.cpp
    docks/RenderInformationDock.cpp
    docks/SceneDock.cpp
    models/OutputSettingsModel.cpp
    models/PPMSettingsModel.cpp
    models/RenderStatisticsModel.cpp
    scene/SceneFactory.cpp
    scene/SceneManager.cpp
    ui/AboutWindow.ui
    ui/ComputeDeviceInformationWidget.ui
    ui/ComputeDeviceInformationWidgetTabPage.ui
    ui/MainWindowBase.ui
    ui/RenderTypePPMInfo.ui
    AboutWindow.cpp
    MainWindowBase.cpp
    RenderWidget.cpp
    ComputeDeviceInformationWidget.cpp
    ComputeDeviceInformationWidgetTabPage.cpp
    Application.cpp
)

add_library(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/Gui_autogen/include") # TODO fix
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Widgets Qt5::Gui RenderEngine ${OPENGL_LIBRARIES})
