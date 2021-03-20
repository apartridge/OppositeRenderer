/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <iostream>
#include <exception>
#include <QApplication>
#include "MainWindowBase.hxx"
#include "StandaloneApplication.h"
#include "ComputeDeviceRepository.h"
#include <QThread>
#include <QTextStream>
#include <QMessageBox>
#include <QPushButton>

int main( int argc, char** argv )
{
    QApplication qApplication(argc, argv);
    qApplication.setOrganizationName("Opposite Renderer");
    qApplication.setApplicationName("Opposite Renderer");

    QTextStream out(stdout);
    QTextStream in(stdin);

    try
    {
        ComputeDeviceRepository repository;

        const std::vector<ComputeDevice> & repo = repository.getComputeDevices();

        out << "Available compute devices:" << endl;

        for(int i = 0; i < repo.size(); i++)
        {
            const ComputeDevice & device = repo.at(i);
            out << "   " <<  i << ": " << device.getName() << " (CC " << device.getComputeCapability() << " PCI Bus "<< device.getPCIBusId() <<")" << endl;
        }

        int deviceNumber = repo.size() == 1 ? 0 : -1;

        while (deviceNumber >= repo.size() || deviceNumber < 0)
        {
            out << "Select 0-" << repo.size()-1 << ":" << endl;
            in >> deviceNumber;
        }

        out << deviceNumber << endl;

        ComputeDevice device = repo.at(deviceNumber);

        StandaloneApplication application(qApplication, device);

        // Run application
       // QThread* applicationThread = new QThread(&qApplication);
      //  application.moveToThread(applicationThread);
      //  applicationThread->start();

        /*QWidget window;
        window.resize(320, 240);
        window.show();
        window.setWindowTitle(
            QApplication::translate("toplevel", "Top-level widget"));

        */


        MainWindowBase mainWindow(application);
        mainWindow.showMaximized();


        int returnCode = qApplication.exec();
        application.wait();

       // applicationThread->quit();
      //  applicationThread->wait();

        return returnCode;
    }
    catch(const std::exception & E)
    {
        QMessageBox::warning(NULL, "Exception Thrown During Launch of Standalone", E.what());
        return -1;
    }
}
