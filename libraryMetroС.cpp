#include "libraryMetro.h"

using namespace std;

void libraryMetroC::findPaperCornels(void* src, int* pointCornels, int width, int height){

    if(flags & ANDROID_OS){

        int rgbSize = width * height * 3;
        rgb = new jbyte[rgbSize];

        NV21_to_RGB(src, rgb, height, width);

        cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        image.data = rgb;

        delete[] rgb;
    }

    if( flags & IPHNONE_OS ){

        cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);

        image.data = CVPixelBufferGetBaseAddress(src);

        CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    }


    vector<vector<cv::Point> > squares;

    cv::Size size(480,640);
        
    resize(image, image, size);
    
    // blur will enhance edge detection
    cv::Mat blurImage(image);
    medianBlur(image, blurImage, 9);
    
    cv::Mat grayImage(blurImage.size(), CV_8U), gray;
    vector<vector<cv::Point> > contours;
    
    // find squares in every color plane of the src
    for (int c = 0; c < 3; c++)
    {
        int ch[] = {c, 0};
        mixChannels(&blurImage, 1, &grayImage, 1, ch, 1);
        
        // try several threshold levels
        const int threshold_level = 2;
        for (int l = 0; l < threshold_level; l++)
        {
            // Use Canny instead of zero threshold level!
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                Canny(grayImage, gray, 10, 20, 3); //
                
                // Dilate helps to remove potential holes between edge segments
                dilate(gray, gray, cv::Mat(), cv::Point(-1,-1));
            }
            else
            {
                gray = grayImage >= (l+1) * 255 / threshold_level;
            }
            
            // Find contours and store them in a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            
            // Test contours
            vector<cv::Point> approx;
            for (size_t i = 0; i < contours.size(); i++)
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(cv::Mat(contours[i]), approx, arcLength(cv::Mat(contours[i]), true)*0.02, true);
                
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(cv::Mat(approx))) > 1000 &&
                    isContourConvex(cv::Mat(approx)))
                {
                    double maxCosine = 0;
                    
                    for (int j = 2; j < 5; j++)
                    {
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
    
    // Find out the biggest square by comparing area of each squares.
    int largest_area = 0;
    int largest_square_index = 0;
    
    for( int i = 0; i< squares.size(); i++ ) // iterate through each square.
    {
        //  Find the area of square
        double a = contourArea( squares[i],false);
        if ( a > largest_area )
        {
            largest_area = a;
            // Store the index of largest square
            largest_square_index = i;
        }
    }
    
    vector<cv::Point> ptsContour;
    ptsContour = squares[largest_square_index];
    
    // sorting coordinat points from top-left in clockwise order
    sort( ptsContour.begin(), ptsContour.end(), SortbyXaxis );
    sort( ptsContour.begin(), ptsContour.end(), SortbyYaxis );
    
    // write coordinates to array
    pointCornels[0] = ptsContour[0].x;
    pointCornels[1] = ptsContour[0].y;
    pointCornels[2] = ptsContour[1].x;
    pointCornels[3] = ptsContour[1].y;
    pointCornels[4] = ptsContour[2].x;
    pointCornels[5] = ptsContour[2].y;
    pointCornels[6] = ptsContour[3].x;
    pointCornels[7] = ptsContour[3].y;
}

void libraryMetroC::clearUpDocument(void* src, int* pointCornels, void* dst, int whithDocument, int heightDocument){

    if(flags & ANDROID_OS){

        int rgbSize = width * height * 3;
        rgb = new jbyte[rgbSize];

        NV21_to_RGB(src, rgb, height, width);

        cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        image.data = rgb;

        delete[] rgb;
    }

    if( flags & IPHNONE_OS ){

        cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);

        image.data = CVPixelBufferGetBaseAddress(src);
        
        CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    }

    cv::Mat paper(heightDocument, whithDocument, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat paperBin(heightDocument, whithDocument, CV_8UC1, cv::Scalar(0));

    // Lambda Matrix
    cv::Mat lambda( 2, 4, CV_32FC1 );
    cv::Mat gray, prevGray, image, frame;

    // Input Quadilateral or Image plane coordinates
    cv::Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    cv::Point2f outputQuad[4];
    
    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = cv::Point2f( 0, 0 );
    outputQuad[1] = cv::Point2f( whithDocument, 0);
    outputQuad[2] = cv::Point2f( whithDocument, heightDocument);
    outputQuad[3] = cv::Point2f( 0, heightDocument);
    
    inputQuad[0] = cv::Point2f( pointCornels[0], pointCornels[1]);
    inputQuad[1] = cv::Point2f( pointCornels[2], pointCornels[3]);
    inputQuad[2] = cv::Point2f( pointCornels[4], pointCornels[5]);
    inputQuad[3] = cv::Point2f( pointCornels[6], pointCornels[7]);
    
    // Get the Perspective Transform Matrix i.e. lambda
    lambda = cv::getPerspectiveTransform( inputQuad, outputQuad );
    
    // Apply the Perspective Transform just found to the src image
    warpPerspective(src,paper,lambda, paper.size() );
        
    cv::Mat paperOutput(paper);
    
    /// Convert to grayscale
    cvtColor( paperOutput, paperOutput, CV_BGR2GRAY );
    
    /// Apply Histogram Equalization
    //equalizeHist( paperOutput, paperOutput );
    
    /// adaptive histogram equalization
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(0.1);
    clahe->apply(paperOutput,paperOutput);
    
    /// Apply Gamma correction
    gammaCorrection(paperOutput, paperOutput, 0.9);
    
    /// Apply filtering image for improving sharpness and unsharp
    //employmentFilter2D(paperOutput, paperOutput);
    cv::Mat paperUnsharp;
    cv::GaussianBlur(paperOutput, paperUnsharp, cv::Size(0, 0), 3);
    cv::addWeighted(paperOutput, 1.5, paperUnsharp, -0.5, 0, paperUnsharp);
    
    /// Binarization using Adaptive thresholding
    adaptiveThreshold(paperOutput, paperBin, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 15, 5); // good result
    //adaptiveThresholdCustom((unsigned char*)paperOutput.data, (unsigned char*)paperBin.data, IMAGE_WIDTH, MAGE_HEIGHT);
        
    cv::Mat element = getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
    
    /// Apply the specified morphology operation
    cv::morphologyEx( paperBin, paperBin, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 1);

    if(flags & ANDROID_OS){
    /// cv::Mat to NV21
    }

    if( flags & IPHNONE_OS ){

        CVPixelBufferLockBaseAddress(dst, kCVPixelBufferLock_ReadOnly);

        CVPixelBufferGetBaseAddress(dst) = paperBin.data;
        
        CVPixelBufferUnlockBaseAddress(dst, kCVPixelBufferLock_ReadOnly);
    }

    
}

void libraryMetroC::NV21_to_RGB(jbyte *frame, jbyte *rgb, int width, int height) {
    jbyte *uv = frame + width * height;
    for (int i = 0; i < width; ++i) {
        if (i & 1) uv -= height;
        int u = 0, v = 0;
        for (int j = 0; j < height; ++j) {
            int y = (BYTE) *frame++ - 16;
            if (y < 0) y = 0;
            if (~j & 1) {
                v = (BYTE) *uv++ - 128;
                u = (BYTE) *uv++ - 128;
            }

            int y1192 = 1192 * y;
            int r = y1192 + 1634 * v;
            int g = y1192 - 833 * v - 400 * u;
            int b = y1192 + 2066 * u;
            *rgb++ = (jbyte) (clamp(r, 0, 0x3FFFF) >> 10);
            *rgb++ = (jbyte) (clamp(g, 0, 0x3FFFF) >> 10);
            *rgb++ = (jbyte) (clamp(b, 0, 0x3FFFF) >> 10);
        }
    }
}

double libraryMetroC::angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 ) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void libraryMetroC::adaptiveThresholdCustom(unsigned char* src, unsigned char* bin, int width, int height)
{
    //    Using method from paper Adaptive Thresholding Using the Integral Image
    //    Derek Bradley, Carleton University, Canada
    //    Gerhard Roth, National Research Council of Canada
    //    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.7883&rep=rep1&type=pdf
    
    unsigned long* integralImage = 0;
    int i, j;
    long sum=0;
    int count=0;
    int index;
    int x1, y1, x2, y2;
    int s2 = S/2;
    
    // create the integral image
    integralImage = (unsigned long*)malloc(width*height*sizeof(unsigned long*));
    
    for (i=0; i<width; i++)
    {
        // reset this column sum
        sum = 0;
        
        for (j=0; j<height; j++)
        {
            index = j*width+i;
            
            sum += src[index];
            if (i==0)
                integralImage[index] = sum;
            else
                integralImage[index] = integralImage[index-1] + sum;
        }
    }
    
    // perform thresholding
    for (i=0; i<width; i++)
    {
        for (j=0; j<height; j++)
        {
            index = j*width+i;
            
            // set the nxn region
            x1=i-s2; x2=i+s2;
            y1=j-s2; y2=j+s2;
            
            // check the border
            if (x1 < 0) x1 = 0;
            if (x2 >= width) x2 = width-1;
            if (y1 < 0) y1 = 0;
            if (y2 >= height) y2 = height-1;
            
            count = (x2-x1)*(y2-y1);
            
            // I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
            sum = integralImage[y2*width+x2] -
            integralImage[y1*width+x2] -
            integralImage[y2*width+x1] +
            integralImage[y1*width+x1];
            
            if ((long)(src[index]*count) < (long)(sum*(1.0-T)))
                bin[index] = 0;
            else
                bin[index] = 255;
        }
    }
    
    free (integralImage);
}

void libraryMetroC::gammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma)
{
    CV_Assert(src.data);
    
    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));
    
    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
    }
    
    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
        case 1:
        {
            
            cv::MatIterator_<uchar> it, end;
            for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
                //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
                *it = lut[(*it)];
            
            break;
        }
        case 3:
        {
            
            cv::MatIterator_<cv::Vec3b> it, end;
            for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
            {
                
                (*it)[0] = lut[((*it)[0])];
                (*it)[1] = lut[((*it)[1])];
                (*it)[2] = lut[((*it)[2])];
            }
            
            break;
            
        }
    }
}

void libraryMetroC::employmentFilter2D(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat kern = (cv::Mat_<char>(3, 3) << -1, -1, -1,
                                    -1, 9, -1,
                                    -1, -1, -1);
    
//    Mat kern = (Mat_<char>(3, 3) << 0,   0.2, 0,
//                                    0.2, 0.2, 0.2,
//                                    0,   0.2, 0);
    
    filter2D(src, dst, src.depth(), kern, cvPoint(-1,-1), 0, cv::BORDER_DEFAULT);
    
}

bool libraryMetroC::SortbyXaxis(const cv::Point & a, const cv::Point &b)
{
    return a.x < b.x;
}
bool libraryMetroC::SortbyYaxis(const cv::Point & a, const cv::Point &b)
{
    return a.y < b.y;
}
