/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/


#pragma once
class QString;
class QFile;
class Image
{
public:
    Image(const QString & imageCompletePath);
    ~Image(void);
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    const unsigned char* constData() const;

private:
    void loadImageFromTga( const QFile & image );
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_depth;
    unsigned char* m_imageData;
};