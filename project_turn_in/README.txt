README.txt

Dataset hosted on uchicago box at https://uchicago.box.com/s/j9ncizfbtn53ipg6to2inwlcjiukzmb0
You should also be shared on this folder.
There are 2 copies of the data, one as pdfs and one as txt files.
all_bills.zip
550MB zipped
800MB unzipped

From the projects folder:

scripts -- A folder with all the code I've written. Note: I apologize if filepaths are misplaced.  I had to restructure how I was saving things as graphs and such proliferated; I did my best to change the filepaths and re-test, but may not always have succeeded.

	web_scraper.py -- the code for my web scraper.  When run, it pulls down all the data from Congress.gov.  Returns bill_sponsorships_final.json and milestones.csv (the latter is just for convenience.

	text_classifier.py -- text classification pipeline.  Built from scratch.  Returns test_output_text_classifier[model].csv
	Drew information from same link as Homework 3: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html.

	build_classifier.py -- general classifier pipeline.  Similar to the one from the Spring Machine Learning class, but I rebuilt it because my old one was returning fairly nonsensical results.  Also added functionality (ROC curve, PR on one line, rebuilding models, etc.).  Returns top_models_from_build_classifier.csv.

	network.py -- Builds the online classifier network from sponsorship data returned by web_scraper.  Generates network_with_senators.csv (prev. version got network.csv)

	rebuild_to_predict.py -- Rebuilds best classifiers from pipelines, integrates predictions and evaluates inbegrated predictions

	Total lines of code in above scripts: ~1,250

	ancillary_scripts -- less important scripts and functions used for random purposes.  Not necessary to understand project; I'm just keeping them here to show some of the stuff I explored, in case that's useful for grading.

		misc_cleaning_functions.py -- functions used to get individual outputs (e.g. common words for presentation slides)

		my_library.py -- useful general functions that I've written throughout the quarter.

		home_for_broken_code.py -- place for old functions that dont work anymore but I sometimes cannibalized or copied.

output -- a folder with the outputs from the scripts

	pipeline_outputs -- ouputs from pipelines

		network.csv, network_with_senators.csv -- centrality measurements from network.py

		output_text_classifier_[model].csv -- models and non-point evaluation metrics from text_classifier.py

		top_models_from_bill_classifier.csv -- models and non-point evaluation metrics from build_classifier.py

	 plots -- graphs and plots 

	 	PR -- folder with various Precision/Recall plots from classifiers.  Number corresponds to number output from output csv.

	 	ROC -- folder with various ROC plots from classifiers.  Number corresponds to number output from output csv. (I names these badly: the ROC #|# ones are from the text classifiers; the others are from the network classifiers)

	 	networks -- folder with various network visualizations from classifiers.

	 milestones -- folder with milestones from webscraper

	 bill_sponsorships_final.json -- ouput from webscraper, bills and sponsorship data

	 passage_data.json -- output from webscraper, bills and if they passed or not

reports_and_slides -- folder with writeups and slides

	AnAnalysisofSenatorialInfluence.pdf -- Initial proposal.

	MidtermReportEstimatingSenatorialInfluence.pdf -- Midterm Report

	RankingSenatorialInfluence.pptx -- Midterm Presentation Slides

	FinalReportPredictingBillPassage -- Final Report.

	PredictingBillPassageFinalPresentation.pptx -- Final Presentation Slides

This project was a (very large) extension of my CS 122 project, which visualized congressional networks.  The code base for that project is at https://github.com/mvasiliou/Congressional-Cosponsor-Relationships (mvasiliou is Mike Vasiliou, one of my partners for that project, along with Komal Kowatra).  The final Django visualization is deployed here: https://warm-crag-35566.herokuapp.com/gov_data/ (it takes ~7 seconds to load).  
IMPORTANT NOTE: I did not take any code or data from my old project.  I wanted to, but we had deleted it when I went back to try to get it.  I just mention it in my initial proposal, so I wanted to give you an idea of what it was.






