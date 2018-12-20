#include "solution.cpp"



typedef BallPosition (*BallFinderMethod)(const char * const fileName, int detail_please);



float radiusPrecision(BallPosition & prediction, BallPosition & correct)
{
	float delta_r = prediction.radius - correct.radius;

	if (delta_r < 0){
		delta_r *= -1;
	}

	return delta_r / correct.radius;
}


float centerPrecision(BallPosition & prediction, BallPosition & correct)
{
	float deltaX = prediction.centerX - correct.centerX;
	float deltaY = prediction.centerY- correct.centerY;

	return sqrtf(deltaX * deltaX + deltaY * deltaY)/ correct.radius;
}



#define MAX_RADIUS_PRECISION 0.25
#define MAX_CENTER_PRECISION 0.25


void processScore(BallFinderMethod method, const char * const trainResults, const char * const testFolder)
{
	FILE * results = fopen(trainResults, "r");

	if (results == NULL){
		cerr << "Couldn't open " << trainResults << " file \n";
		exit(EXIT_FAILURE);
	}

	char file_name[512];
	size_t folerNameSize = strlen(testFolder);

	strcpy(file_name, testFolder);
	file_name[folerNameSize] = '/';

	BallPosition correctPos;

	unsigned totalNbTests = 0;
	unsigned nbTestsWithBall = 0;


	unsigned nbTreated = 0;
	unsigned nbTestsWithBallButNotFound = 0;
	unsigned nbTestsWithNoBallButFound = 0;

	unsigned nbPassed = 0;

	float radPrecSum = 0.;
	float centerPrecSum = 0.;
	float maxRadPrec = 0.;
	float maxCenterPrec = 0.;


	while (EOF != fscanf(results, "%s %d %f %f %f\n",
			file_name + folerNameSize + 1,
			&(correctPos.is_found), &(correctPos.centerX),
			&(correctPos.centerY), &(correctPos.radius))){

		totalNbTests++;

		if (correctPos.is_found){
			nbTestsWithBall++;
		}

		BallPosition predictedPos = method(file_name, 0);

		if (!predictedPos.is_found){

			if (correctPos.is_found){
				nbTestsWithBallButNotFound++;
				cout << file_name << " !!!\n";
				method(file_name, 1);
			}

			continue;

		} else if (!correctPos.is_found){
			nbTestsWithNoBallButFound++;
			cout << file_name << " FALSE POSITIVE !!!!!!\n";
			method(file_name, 1);
			continue;
		}


		nbTreated++;

		float radPrec = radiusPrecision(predictedPos, correctPos);
		float centerPrec = centerPrecision(predictedPos, correctPos);

		if (maxRadPrec < radPrec){
			maxRadPrec = radPrec;
		}

		if (maxCenterPrec < centerPrec){
			maxCenterPrec = centerPrec;
		}

		if (radPrec > MAX_RADIUS_PRECISION || centerPrec > MAX_CENTER_PRECISION){
			cout << file_name << " " << radPrec << ", " << centerPrec << "\n";
			method(file_name, 1);
		} else {
			nbPassed++;
		}

		radPrecSum += radPrec;
		centerPrecSum += centerPrec;
	}

	float avgRadPrec = nbTreated ? radPrecSum / nbTreated : 0.;
	float avgCenterPrec = nbTreated ? centerPrecSum / nbTreated : 0.;


	cout << "\n\n\tRESULTS\n\n";

	cout << "totalNbTests : " << totalNbTests << "\n\n";

	cout << "nbTestsWithBall : " << nbTestsWithBall << "\n";
	cout << "nbTestsWithBallButNotFound : " << nbTestsWithBallButNotFound <<"\n";
	cout << "nbTestsWithNoBallButFound : " << nbTestsWithNoBallButFound <<"\n\n";

	cout << "nbTreatedBallScores : " << nbTreated << "\n";
	cout << "nbScorePassed : " << nbPassed << "\n\n";

	cout << "avgRadPrec : " << avgRadPrec << "\n";
	cout << "avgCenterPrec : " << avgCenterPrec << "\n\n";

	cout << "maxRadPrec : " << maxRadPrec << "\n";
	cout << "maxCenterPrec : " << maxCenterPrec << "\n";

	fclose(results);
}





int main(int argc, char * argv[])
{
	if (argc != 3){
		cerr << "Usage : " << argv[0] << " trainResults testFolder\n";
		exit(EXIT_FAILURE);
	}

	processScore(ultimateSol, argv[1], argv[2]);

	return 0;
}
