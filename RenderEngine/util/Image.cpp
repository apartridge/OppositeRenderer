/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Image.h"
#include <QFileInfo>
#include <QFile>
#include <QString>
#include <QImage>
#include "imageformats/libtga/tga.h"

Image::Image(const QString & imageCompletePath)
    : m_width(0),
    m_height(0),
    m_depth(0),
    m_imageData(NULL)
{
    QFile imageFile (imageCompletePath);

    if(!imageFile.exists())
    {
        QString string = QString("The file %1 does not exist.").arg(imageCompletePath);
        throw std::runtime_error(string.toStdString());
    }

    if(imageFile.open(QIODevice::ReadOnly))
    {
        QFileInfo fileInfo(imageFile);
        QString imageExtension = fileInfo.suffix().toLower();
        if(imageExtension == "tga")
        {
            loadImageFromTga(imageFile);
        }
        else
        {
            QImage image(imageCompletePath);
            if(!image.isNull())
            {
                QImage convertedImage = image.convertToFormat(QImage::Format_ARGB32);
                if(convertedImage.isNull())
                {
                    QString string = QString("Image is null.");
                    throw std::runtime_error(string.toStdString());
                }
                else
                {
                    const uchar* data = convertedImage.constBits();
                    m_imageData = new unsigned char[convertedImage.width()*convertedImage.height()*4];
                    for(int i = 0; i < convertedImage.width()*convertedImage.height()*4; i+=4)
                    {
                        m_imageData[i+0] = data[i+1];
                        m_imageData[i+1] = data[i+2];
                        m_imageData[i+2] = data[i+3];
                        m_imageData[i+3] = data[i+0];
                    }
                }
            }
            else
            {
                QString error = QString("Unable to load image %1 using Qt.").arg(imageCompletePath);
                throw std::runtime_error(error.toStdString());
            }
        }
    }
    else
    {
        QString string = QString("An error occurred trying to open the image %1.").arg(imageFile.fileName());
        throw std::runtime_error(string.toStdString());
    }
}

Image::~Image(void)
{
    delete m_imageData;
}

void Image::loadImageFromTga( const QFile & image )
{
    FILE* filePtr = fdopen(image.handle(), "r");
    if(filePtr != NULL)
    {
        TGA* tgaHandle = TGAOpenFd(filePtr);
        TGAData tgaData;
        tgaData.flags = TGA_IMAGE_DATA | TGA_RGB; // We set what we want back from TGAReadImage

        int tgaReadImageReturnCode = TGAReadImage(tgaHandle, &tgaData);
        if(tgaReadImageReturnCode == TGA_OK)
        {
            if( (tgaHandle->hdr.img_t == 2 || tgaHandle->hdr.img_t == 3) && tgaHandle->hdr.map_t == 0)
            {
                m_width = tgaHandle->hdr.width;
                m_height = tgaHandle->hdr.height;
                m_depth = tgaHandle->hdr.depth/8;
                m_imageData = new unsigned char[m_width*m_height*4];

                bool flipY = (tgaHandle->hdr.y == 0);

                for(int y = 0; y < m_height; y++)
                {
                    int yOut = y; //flipY ? m_height - y - 1 : y;
                    for(int x = 0; x < m_width; x++)
                    {
                        m_imageData[(yOut*m_width+x)*4 + 0] = tgaData.img_data[(y*m_width+x)*m_depth + 0];
                        m_imageData[(yOut*m_width+x)*4 + 1] = tgaData.img_data[(y*m_width+x)*m_depth + 1];
                        m_imageData[(yOut*m_width+x)*4 + 2] = tgaData.img_data[(y*m_width+x)*m_depth + 2];
                        m_imageData[(yOut*m_width+x)*4 + 3] = 1;
                    }
                }

                free(tgaData.img_data);
            }
            else
            {
                QString error;
                if(tgaHandle->hdr.map_t == 1)
                {
                    error = QString("Does not support TGA color map");
                }
                else
                {
                    error = QString("Only supports RGB(A) type TGA textures - does not support type '%1'")
                        .arg(tgaHandle->hdr.img_t);
                }

                error += QString(" for %1").arg(image.fileName());
                throw std::runtime_error(error.toStdString());
            }
        }
        else
        {
            QString error = QString("Unknown error reading TGA data for %1.").arg(image.fileName());
            throw std::runtime_error(error.toStdString());
        }
    }
    else
    {
        QString error = QString("FILE*=null for TGA file %1.").arg(image.fileName());
        throw std::runtime_error(error.toStdString());
    }

}

unsigned int Image::getWidth() const
{
    return m_width;
}

unsigned int Image::getHeight() const
{
    return m_height;
}

const unsigned char* Image::constData() const
{
    return m_imageData;
}
