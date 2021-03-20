project(Server LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(
    SOURCES
    gui/ui/ReadyForRenderingWidget.ui
    gui/ui/ServerWindow.ui
    gui/ui/SetServerSettingsWidget.ui
    gui/ReadyForRenderingWidget.cpp
    gui/ServerWindow.cpp
    gui/SetServerSettingsWidget.cpp
    gui/WaitingForConnectionWidget.cpp
    server/RenderServer.cpp
    server/RenderServerRenderer.cpp
    server/RenderServerState.cpp
    server/RenderServerState.cpp
    server/ServerState.cpp
    main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network RenderEngine Gui)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)