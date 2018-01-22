# HDFC-Stock-Model
This is a Machine Learning model developed with "Random Forest Algorithm" to predict the turnover of HDFC bank with a given dataset of the previous turnovers and features.
The dataset is real taken from QUANDL.

Quandl Code: NSE/HDFCBANK 

If you have the Quandle Python package already installed, you can use the following link: quandl.get("NSE/HDFCBANK")

No feature scaling used as Random Forest does it internally.

No cross-validation used as the training set already had 43 entries. Since we had to predict only one entry, we would get a more accurate result.

Results:

Actual Turnover(dated 18th January, 2018): ₹16654

Predicted Turnover(dated 18th January, 2018): ₹17044

**All predictions are based on that day's features.**
