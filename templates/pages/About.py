import streamlit as st

st.set_page_config(
    page_icon="🌦️"
)

st.title("Project Info")
st.divider()

st.write("The project is meant to implement production level code structure, best practises and production tools and strategies. "
        "Building a highly accurate model is not the primary goal of this project. Building an end-to-end ML project is "
        "significantly complex and difficult task, this project implementation would reavel the challenges and hidden works "
        "that real world ML project possess.")

st.subheader("Project Development Status")
st.markdown("1. Created the production level ML template  ✅\n"
"2. Added commonly used custom functions  ✅\n"
"3. Wrote modular code for data ingestion steps ✅\n"
"4. Implemented pipeline for data ingestion ✅\n"
"5. Wrote modular code for data validation steps ✅\n"
"6. Implemented pipeline for data validation ✅\n"
"7. Implemented pipeline for data transformation ✅\n"
"8. Implemented pipeline for model training ✅\n"
"9. Implemented pipeline for model evaluation ✅\n"
"10. Model deployment ✅\n"
"11. Model performance enhancement `Working` \n")

st.subheader("Data Collection")
st.markdown("Data is collected from [Met Office API](https://www.metoffice.gov.uk/services/data/met-office-weather-datahub) "
            "3 hours spot data is collected through the API, only 24 samples are available since the API doesn't support "
            "large data download at a time. So, there is a seperate GitHub [repository](https://github.com/jayasooryantm/weather-data) "
            "created for data collection which integrated with github action to automate the data collection, every day "
            "the data is collected through the API and saved as json file, for ease of use the script will zip the "
            "entire data and save it in the repo. This zip file is used as project data.")

st.subheader("Data Transformation & Cleaning")
st.markdown("There is significant work have been done to clean and transform the data which is used to train the model. "
            "Since the data is in json format it needs to be changed into csv format for training. Once the zip file "
            "is extracted each json file is read and chunk of data (24 samples per file) is converted into dataframe "
            "and the related features are added separately to dataframe, each json file is turned into dataframe and "
            "appended to the previous one. In the cleaning stage headers of the dataframe is changed then dropped null "
            "values (even though there are none found so far), few features are not numbers in the dataframe, string "
            "values are converted to numbers using dictionary mapping since the values are limited. date column is "
            "converted as datetime then split into separate columns as day, month, year. After these operations there "
            "is a validation check, check the column names against the stored column name then check the data type of "
            "the column against the stored data type of each column, which make sure the data is in good shape. The "
            "validation process will create a text file with a status (True or False) if the validation failed, "
            "process won't run any further.")

st.subheader("Model Training")
st.markdown("PyTorch framework is used for the development of the model, since the project focusing on replicating "
            "real-world production level implementation, I have chosen neural network for the problem irrespective of "
            "model performance. The goal is to build a system that resembles real-world ML project and face the "
            "challenges on the way, and eventually make the model better, make the project perfect. I have used "
            "multi-output neural network model which could predict all 6 parameters at a time. Model comprises one "
            "shared layer and 6 individual layers for each feature prediction. The model evaluated with 4 metrics "
            "which are MSE, RMSE, MAE and R2. Currently the model doesn't produce any insightful result, but model "
            "enhancement is progressing.")

st.subheader("Tools and Frameworks")
st.markdown("**Python: 3.9**  \n"
            "The entire project is made using python (Which is obvious)")
st.markdown("**PyTorch: 2.0.1**  \n"
            "Choice of ML framework is PyTorch because it is more pythonic and easy to code, also the code structure "
            "make sense what is happening under-the-hood. Unlike the TensorFlow framework it doesn't have heavy "
            "encapsulation which gives developer an idea of what's going on.")
st.markdown("**Streamlit** [Deployment] \n"
            "For the deployment I have chosen the easy way, which is [Streamlit](https://streamlit.io/). This make the "
            "deployment so easy that I don't need to worry about the server side bottlenecks. Also it supports "
            "continuous integration from GitHub, once I commit changes into GitHub it will reflect in server with no "
            "time")
st.markdown("**MLFlow** [Experiment Tracking] \n"
            "It is important to keep track of the experiment and the artifact, it is necessary for comparison of the "
            "model and performance. So I have chosen one of the simple and best out there, MLFlow. Hyper-Parameters "
            "and training metrics are store in the MLFlow server.")
st.markdown("**DagsHub** [MLFlow Hosting] \n"
            "DagsHub is a open source data science collaboration platform. Which consist many features but I am using "
            "only the experiment tracking feature from it, MLFlow is hosted in "
            "[DagsHub](https://dagshub.com/jayasooryan.tm/weather-prediction/experiments/).")
st.markdown("**Project Structure**  \n"
            "The below project structure is used to make the code repository more usable and easy to debug. Modular "
            "code is written and modules are imported to pipeline file, a pipeline class is made to seamless and "
            "sequential processing of the app. The entire code is uploaded in [GitHub](https://github.com/jayasooryantm/weather-prediction)")
st.image("templates/pages/Project_Structure.png", width=500)