# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

# Utils
import base64 

#Data Viz Library
from pandas_profiling import ProfileReport
import sweetviz as sv
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import codecs

def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


# Standard Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 


def csv_downloader(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "Processed_data_file.csv"
    st.markdown("#### Download File ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
    st.markdown(href,unsafe_allow_html=True)

    
def main():
	"""Semi Automated ML App with Streamlit """

	activities = ["EDA(basic)","Pandas Profile","Sweetviz","Plots","Preporcess Data"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	st.set_option('deprecation.showPyplotGlobalUse', False)
	if choice == 'EDA(basic)':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			all_columns = df.columns.to_list()
			numeric_data = df.select_dtypes(include=[np.number])
			numbers = numeric_data.columns.to_list()
			categorical_data = df.select_dtypes(exclude=[np.number])
			categories=categorical_data.columns.to_list()

			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)
			    
			if st.checkbox("Show Data Types"):
			    st.write(df.dtypes)
			    
			if st.checkbox("Null value count"):
			    st.write(df.isnull().sum())   
			
			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Value Counts For categorical values"):
			    selected_column = st.selectbox("Select a Column",categories)
			    st.write(df.loc[:,selected_column].value_counts())
			    
			if st.checkbox("Correlation Plot(Matplotlib)"):
			    
				plt.matshow(df.corr())
				st.pyplot()

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()


			if st.checkbox("Pie Plot"):
				all_columns = df.columns.to_list()
				column_to_plot = st.selectbox("Select 1 Column",all_columns)
				pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()
  
	elif choice == "Sweetviz":
		st.subheader("Automated EDA with Sweetviz")
		data_file = st.file_uploader("Upload CSV",type=['csv'])
		if data_file is not None:
			df = pd.read_csv(data_file)
			st.dataframe(df.head())
			if st.button("Generate Sweetviz Report"):
				report = sv.analyze(df)
				report.show_html()
				st_display_sweetviz("SWEETVIZ_REPORT.html")

	elif choice == "Pandas Profile":
	    st.subheader("Automated EDA with Pandas Profile")
	    data_file = st.file_uploader("Upload CSV",type=['csv'])
	    if data_file is not None:
	        df = pd.read_csv(data_file)
	        st.dataframe(df.head())
	        profile = ProfileReport(df)
	        st_profile_report(profile)



	elif choice == 'Plots':
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
            
			# Customizable Plot

			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()


	
	elif choice == 'Preporcess Data':
		st.subheader("Data Preprocessing")
		data = st.file_uploader("Upload a Dataset", type=["csv"])
		if data is not None:
			df = pd.read_csv(data)
			all_columns = df.columns.to_list()
			numeric_data = df.select_dtypes(include=[np.number])
			numbers = numeric_data.columns.to_list()
			

			    
			if st.checkbox("Remove NA values"):
			    st.success("Droped all NA values")
			    df.dropna(inplace=True)
			    #st.dataframe(df.head())
			    
			if st.checkbox("Normalize data"):
			    st.markdown("### Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1")
			    st.write("Note : Only use when data does not follow gaussian distribution.")
			    
			    selected_columns = st.multiselect("Select Columns to Normalize",numbers)
			    st.info("Select the columns which you want to Normalize")
			    if st.button("Yes Normalize Please"):
			        new_df=df[selected_columns]
			        df1= df.drop(selected_columns,axis=1)
			        normalized_df=(new_df-new_df.min())/(new_df.max()-new_df.min())
			        result = df1.join(normalized_df)
			        
			        csv_downloader(result)
			    
			    
			    
			if st.checkbox("Standardize Data"):
			     st.markdown("### Standardization is a scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.")
			     st.write("Note : Only use when data follows gaussian distribution. But not necessarily true.")
			     selected_columns = st.multiselect("Select Columns to Standardize",numbers)
			     st.info("Select the columns which you want to Standardize")
			     if st.button("Yes Standardize Please"):
			         new_df=df[selected_columns]
			         df1= df.drop(selected_columns,axis=1)
			         normalized_df=(new_df-new_df.mean())/new_df.std()
			         result = df1.join(normalized_df)
			         csv_downloader(result)
			     
			     
			     
			    
            
			


    
        		  
if __name__ == '__main__':
	main()
