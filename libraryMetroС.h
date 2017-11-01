#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <ctype.h>

typedef unsigned char BYTE;

enum LibraryOSFlags {
       ANDROID_OS    = 0, //!< Use this value for Android OS: set to 0.
       IPHNONE_OS    = 1, //!< Use this value for iOS: set to 1.
};

class libraryMetroC{
	public:
		/// Main methods for SDK and application

		static void findPaperCornels(void* src, int* pointCornels, int width, int height, int flags);

		/** @brief Recognition the white paper document.

		The function finds squares and outputs corners coordinates of the biggest square.

		@param src Source 8-bit 3-channel RGB image.
		@param pointCornels Array of corners points from top-left in clockwise order.
		@param width Size of the input image.
		@param height Size of the input image.
		@param flags Flag that can take values of LibraryOSFlags.

		 */

		static void clearUpDocument(void* src, int* pointCornels, void* dst, int widthDocument, int heightDocument, int flags );

		/** @brief Applies an filtering and unwarpping of the image.

		The function processes perspective transformation and filtering src image.

		@param src Source 8-bit 3-channel RGB image.
		@param pointCornels Array of coordinat points from top-left in clockwise order.
		@param dst Destination image of the same size and the same type as src.
		@param widthDocument Width size of the outpute dst image.
		@param heightDocument Height size of the outpute dst image.
		@param flags Flag that can take values of LibraryOSFlags.

		 */

		//////////////////////////////////////////////
	private:

		static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 );
		static void NV21_to_RGB(jbyte *frame, jbyte *rgb, int width, int height);
		static bool SortbyXaxis(const cv::Point & a, const cv::Point &b);
		static bool SortbyYaxis(const cv::Point & a, const cv::Point &b);
		static void employmentFilter2D(cv::Mat& src, cv::Mat& dst);
		static void gammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);
		static void adaptiveThresholdCustom(unsigned char* src, unsigned char* bin, int width, int height);

};		
