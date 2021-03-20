project(Client LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(
    SOURCES
    client/commands/GetServerDetailsCommand.cpp
    client/commands/ServerCommand.cpp
    client/commands/ServerCommandResult.cpp
    client/RenderResultPacketReceiver.cpp
    client/RenderServerConnection.cpp
    client/RenderServerConnections.cpp
    client/RenderServerState.cpp
    gui/dialogs/AddNewServerConnectionDialog.cpp
    gui/dialogs/ui/AddNewServerConnectionDialog.ui
    gui/docks/ui/ConnectedServersDock.ui
    gui/docks/ui/RenderServersSummaryDock.ui
    gui/docks/ConnectedServersDock.cpp
    gui/docks/RenderServersSummaryDock.cpp
    gui/ClientMainWindow.cpp
    gui_models/ConnectedServersTableModel.cpp
    DistributedApplication.cpp
    main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network RenderEngine Gui)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)