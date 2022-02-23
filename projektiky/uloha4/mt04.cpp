#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>




//  Funkce pro generování bílého šumu.
//  Intenzita udává maximální velikost šumu.
void bilySum( const cv::Mat& src, cv::Mat& dst, float intenzita )
{
	cv::RNG rng;
	cv::Mat noise = cv::Mat::zeros( src.size(), CV_32SC1 );
	for( int i = 0; i < noise.rows*noise.cols; ++i ) {
		// vygenerujte hodnoty pro bílý šum a uložte do obrazu se šumem
		// využijte: rng.gaussian(double sigma)
		// a vztah pro gaussův pásmový bílý šum
		// vzorky bílého šumu s výkonem (rozptylem) např. 0.2 vypočteme:
		// v = sqrt(výkon=0.2) * gaussian(sigma=1.0)
		// funkce gaussian() generuje náhodné vzorky s normálním rozlošením se střední hodnotou 0 a rozptylem 1.
		noise.at<int>(i) = sqrt(intenzita) * rng.gaussian(1);
	}
	cv::Mat tmp;
	src.convertTo( tmp, noise.type() );
	tmp = tmp+noise;
	tmp.convertTo( dst, src.type() );
}




//  Funkce pro filtraci obrazu Gaussovým filtrem.
//     - velikost filtru v pixelech
//	   - sigma je rozptyl pro výpočet hodnot Gaussovy funkce
void separabilityGauss( const cv::Mat& src, int velikost, double sigma, cv::Mat& sepDst, cv::Mat& noSepDst, int &noSepCnt, int &sepCnt )
{
	// velikost - chceme liché číslo, minimálně 3
	int stred = velikost/2;
	stred = MAX(1,stred);
	velikost = 2*stred+1;

	// připravte Gaussův filtr v 1D 
	cv::Mat gauss1D = cv::Mat::zeros( 1, velikost, CV_64FC1 );
	
	// implementujte výpočet koeficientù Gaussova filtru - ručně, dosazením do Gaussovy funkce
	/*  Working area - begin */
	// generate gabor kernel
	for (int k = - stred; k <= stred; k++)
	{
		double sample = exp(-(k * k) / (2 * sigma * sigma));
		double normalizer = sqrt(2 * CV_PI) * sigma;
		gauss1D.at<double>(0, k + stred) = sample / normalizer;
	}
	/*  Working area - end */
	// normalizace hodnot
	gauss1D = gauss1D / (sum(gauss1D).val[0]+0.00001);

	// připravíme Gaussův filtr ve 2D 
	// využijeme konvoluce 1D Gauss. jádra ve směru x a y s jednotkovým impulsem
	cv::Mat gauss2D = cv::Mat::zeros( velikost, velikost, CV_64FC1 );
	gauss2D.at<double>(stred,stred) = 1.;
	filter2D( gauss2D, gauss2D, -1, gauss1D );
	filter2D( gauss2D, gauss2D, -1, gauss1D.t() );
	gauss2D = gauss2D / (sum(gauss2D).val[0]+0.00001);

	// aplikujte připravený filtr na zdrojový obraz
	// využijte separability operátoru - využít 1D filtr
	// implementujte s využitím opencv funkce 'filter2D' a transpozice matice mat.t()
	/*  Working area - begin */
	cv::filter2D(src, sepDst, -1, gauss1D.t());
	cv::filter2D(sepDst, sepDst, -1, gauss1D);
	/*  Working area - end */


	// nyní aplikujte připravený 2D filtr na zdrojový obraz
	// implementujte s využitím opencv funkce 'filter2D'
	/*  Working area - begin */
	cv::filter2D(src, noSepDst, -1, gauss2D);
	/*  Working area - end */

	// ručně spočtěte a uložte do proměnných - počet operací pro výpočet hodnoty 
	// jednoho pixelu výsledného obrazu s/bez využití separability 
	// stačí zjednodušený výpočet - počet operací násobení na jeden pixel

	////////////////////////////////////////////////////////////////////////////////////////////////
	// nezalezi na vyuziti separability operatoru aj tak oba pristupi musia mat rovnaky pocet     //
	// nasobeni, pretoze gauss2D vnikol z gauss1D * gauss1d.t(), cize v konecnom dosledku pri     //
	// aplikacii filtru mam rovnaky pocet nasobeni na jeden pixel.                                //
	////////////////////////////////////////////////////////////////////////////////////////////////
	sepCnt   = gauss1D.cols * cv::Mat(gauss1D.t()).rows;
	noSepCnt = gauss2D.cols * gauss2D.rows;
	
	return;
}



/* Vyhodnocení/porovnání výsledku s referenčním obrazem. */
void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save = false);


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//
// Examples of input parameters
// ./mt-04 ../../data/garden.png 0.2 31 2.0

int main(int argc, char* argv[])
{
    std::string img_path = "";
	float noise_amp = 0.1;
    int filter_size = 45;
	float sigma = 8.0;

	// check input parameters
	if( argc > 1 ) img_path = std::string( argv[1] );
	if( argc > 2 ) noise_amp = atof( argv[2] );
	if( argc > 3 ) filter_size = atoi( argv[3] );
	if( argc > 4 ) sigma = atof( argv[4] );

	// load testing images
	cv::Mat src_rgb = cv::imread( img_path );

	// check testing images
	if( src_rgb.empty() ) {
		std::cout << "Failed to load image: " << img_path << std::endl;
		return -1;
	}

	cv::Mat src_gray;
	cv::cvtColor( src_rgb, src_gray, cv::COLOR_BGR2GRAY );

	//---------------------------------------------------------------------------

	cv::Mat diff, gray, zasum, sepOut, noSepOut, gauss_ref;
	int noSepCnt, sepCnt;
	
	bilySum( src_gray, zasum, noise_amp );
	
	separabilityGauss( zasum, filter_size, sigma, sepOut, noSepOut, noSepCnt, sepCnt );
	cv::GaussianBlur( zasum, gauss_ref, cv::Size(filter_size,filter_size), sigma );

	// vyhodnocení
	checkDifferences( zasum,     src_gray, "04_noise", true );
	checkDifferences( sepOut,   gauss_ref, "04_gaussSep", true );
	checkDifferences( noSepOut, gauss_ref, "04_gaussNoSep", true );
	std::cout << " " << noSepCnt << " " << sepCnt << std::endl;

    return 0;
}
//---------------------------------------------------------------------------




void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save )
{
	double mav = 255., err = 255., nonzeros = 255.;
	cv::Mat diff;

	if( !test.empty() && !ref.empty() ) {
		cv::absdiff( test, ref, diff );
		cv::minMaxLoc( diff, NULL, &mav );
		err = ( cv::sum(diff).val[0] / (diff.rows*diff.cols) );
		nonzeros = 1. * cv::countNonZero( diff ) / (diff.rows*diff.cols);
	}
	
	if( save ) {
		if( !test.empty() ) { cv::imwrite( (tag+".png").c_str(), test ); }
		if( !diff.empty() ) { diff *= 255;	cv::imwrite( (tag+"_err.png").c_str(), diff ); }
	}

	printf( "%s %.2f %.2f %.2f ", tag.c_str(), err, nonzeros, mav );

	return;
}

