# README

**Details As Directed**  
   - **Dataset**: Where it’s from, any steps to load it.
     - The dataset is from "https://raw.githubusercontent.com/peetck/IMDB-Top1000-Movies/refs/heads/master/IMDB-Movie-Data.csv", the code itself allows for sub-sampling, but the "test_movie_data_500.csv" in the repository is from that link but just sampled and saved that can also be run. The code by default, runs on the above dataset.
   - **Setup**: Python version, virtual environment instructions, and how to install dependencies (`pip install -r requirements.txt`).
     - Python setup is standard, the version I used was 3.10.4 direct. Dependencies are installed by the usual pip install methods, i.e.,
       - pip install numpy
       - pip install pandas
       - pip install scikit-learn
   - **Running**: How to run your code (e.g., `python recommend.py "Some user description"` or open your notebook in Jupyter).
     - In the terminal (I used VSCode), enter 'py recommend.py "Some user description"'
     - You can also input other info via --top_n TOP_N and --data_loc DATA_LOC.
   - **Results**: A brief example of your system’s output for a sample query.
     -  Based on "test_movie_data_500.csv", and using the query, "I love thrilling action movies set in space, with a comedic twist." An example output would be:
       - ['1. Star Trek Beyond (Score: 0.2613)',
          '2. Independence Day: Resurgence (Score: 0.2489)',
          '3. Tomorrowland (Score: 0.2294)']  
   - **Salary expectation per month (Mandatory)**:
     - Assuming 20 hours a week with a hourly rate of $20, ~$1600 per month.
