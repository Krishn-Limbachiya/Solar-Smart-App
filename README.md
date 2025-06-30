Step-by-Step Setup Guide for set up SolarSmart with Python 3.13:

**Step 1:** Check Your Python Version

bashpython --version
# Should show Python 3.13.x

**Step 2:** Create a Virtual Environment
bash# Create virtual environment
python -m venv solarsmart_env

# Activate it On Windows:
solarsmart_env\Scripts\activate

**Step 3:** Upgrade pip
bashpython -m pip install --upgrade pip

**Step 4:** Install Dependencies
Save the updated requirements.txt file and install:
bashpip install -r requirements.txt

**Step 5:** Create the Main Application File
Save the Python code: solarsmart_app.py

**Step 6:** Run the Application
bashstreamlit run solarsmart_app.py
