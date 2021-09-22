from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_predict
import pandas as pd
from ctgan import CTGANSynthesizer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor


# function to plot target proportion for binary class in classification dataset
def target_proportion_plot(datafile):
    plt.figure()
    ax7=plt.axes()
    target = datafile["Treatment"]
    plt.title('Target repartition')
    ax7.set(xlabel='Default proportion')
    target.value_counts().plot.pie()
    plt.show()


# function to encode the categorical variables using auto labels
def label_encode(datafile, column_name):
    labelencoder = LabelEncoder()
    datafile[column_name] = labelencoder.fit_transform(datafile[column_name])


# function to augment data using CT-GANSynthesizer
def Ctgan_augment(datafile, n_samples, iterations):
    final_df = datafile
    for i in range(0, iterations):
        ctgan = CTGANSynthesizer()
        ctgan.fit(datafile, datafile.columns, epochs=5)

        samples = ctgan.sample(n_samples)

        print(type(samples))
        final_df = samples.append(final_df, ignore_index= True)
    return final_df


# function to augment data using combination of under-sampling and SMOTE and return the augmented dataset
def num_data_augment(datafile, n_samples):
    final_df = pd.DataFrame()
    rows = 0
    # label encoding Species variable
    label_encode(datafile, 'Species')
    # initiating while loop to generate rows upto 240
    while rows < 240:
        # creating feature set by removing response variable
        features = datafile.drop(['Treatment'], axis=True)

        # using SMOTE to balance the train data
        balance = SMOTE()
        feature, target = balance.fit_resample(features, datafile['Treatment'])
        balanced_df = pd.concat([feature, target.reset_index(drop=True)], axis=1)

        # selecting samples to temporary remove from training set
        samples = balanced_df[balanced_df['Treatment'] == 'Frozen'].sample(n_samples)
        undersampled = samples.append(balanced_df[balanced_df['Treatment'] == 'Lyophilized'], ignore_index=True)
        features = undersampled.drop(['Treatment'], axis=True)

        # using SMOTE to balance the data
        feature, target = balance.fit_resample(features, undersampled['Treatment'])
        balanced_df = pd.concat([feature, target.reset_index(drop=True)], axis=1)

        # appending the samples previously removed.
        temp = balanced_df.append(datafile[datafile['Treatment'] == 'Frozen'], ignore_index=True)
        temp.drop_duplicates(inplace=True)

        features = temp.drop(['Treatment'], axis=True)

        # using SMOTE to balance the data
        feature, target = balance.fit_resample(features, temp['Treatment'])
        balanced_df = pd.concat([feature, target.reset_index(drop=True)], axis=1)

        #appending the data generated in each iteration to the final dataframe
        final_df = final_df.append(balanced_df, ignore_index=True)
        rows = final_df.shape[0]

    return final_df


# function to plot histogram and density plot in single plot.
def hist_density_plot(datafile1, datafile2):
    label_encode(datafile1, 'Species')
    label_encode(datafile2, 'Species')
    label_encode(datafile1, 'Treatment')
    label_encode(datafile2, 'Treatment')
    for i in datafile1.columns:
        f, axs = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
        sns.distplot(datafile1[i], ax = axs[0])
        plt.title("Augmented Data Dist")

        sns.distplot(datafile2[i], ax = axs[1])
        plt.title("Original Data Dist")
        plt.show()


# function to plot spectrometry data row-wise
def plot_spectrometry(datafile1, datafile2):
    orig = datafile1.drop(['Treatment','Species'], axis=1)
    orig.T.plot()
    plt.title('original')

    aug = datafile2.drop(['Treatment','Species'], axis=1)
    aug.T.plot()
    plt.title('augmented')
    plt.show()


# function to split data into training and test set and return train set and test set
def data_split(datafile):
    feature = datafile.drop(['Treatment'], axis=True)
    target = datafile['Treatment']

    train, test, target_train, target_test = train_test_split(feature, target, test_size = 0.40, random_state = 735)

    train_datafile = pd.concat([target_train.reset_index(drop=True), train.reset_index(drop=True)], axis=1)
    test_datafile = pd.concat([target_test.reset_index(drop=True), test.reset_index(drop=True)], axis=1)

    return train_datafile, test_datafile


# function to apply PCA and generate principal components and return pca instance, PC datafile, PCA transformed test data
def pca(traindata, n_comp, testdata):
    feature = traindata.drop(['Treatment'], axis=True)
    target = traindata['Treatment']

    test_feature = testdata.drop(['Treatment'], axis=True)
    test_target = testdata['Treatment']

    scaler = StandardScaler()
    scaler.fit(feature)
    features_scaled = scaler.transform(feature)
    test_features_scaled = scaler.transform(test_feature)

    pca = PCA(n_comp)
    pca.fit(features_scaled)
    principalcomponents = pca.transform(features_scaled)
    testcomp = pca.transform(test_features_scaled)

    principaldf = pd.DataFrame(data=principalcomponents)
    testdf = pd.DataFrame(data=testcomp)

    finaldf = pd.concat([principaldf, target.reset_index(drop=True)], axis=1)
    testdata = pd.concat([testdf, test_target.reset_index(drop=True)], axis=1)

    return pca, finaldf, testdata

# function to apply PLS to generate components and return components, PLS transformed test set, predicted response values
def pls(traindata, testdata, n_comp):
    label_encode(traindata, 'Treatment')
    label_encode(testdata, 'Treatment')

    feature = traindata.drop(['Treatment'], axis=True)
    y_train = traindata['Treatment']

    test_feature = testdata.drop(['Treatment'], axis=True)
    y_test = testdata['Treatment']

    pls_DA = PLSRegression(n_components=n_comp)
    pls_DA.fit(feature, y_train)

    scores_train = pd.DataFrame(pls_DA.x_scores_)

    pls_DA.fit(test_feature, y_test)
    y_pred = pls_DA.predict(test_feature)

    print('confusion matrix for PLS-DA:\n', confusion_matrix(y_test, y_pred.round()))
    print('evaluation metrics for PLS-DA:\n', classification_report(y_test, y_pred.round()))

    scores_test = pd.DataFrame(pls_DA.x_scores_)

    scores_traindf = pd.concat([scores_train, traindata['Treatment'].reset_index(drop=True)], axis=1)
    scores_testdf = pd.concat([scores_test, testdata['Treatment'].reset_index(drop=True)], axis=1)

    return scores_traindf, scores_testdf, y_pred


# function to plot scores of top 2 components generated by PLS
def scores_plot(scores_train):
    feature = scores_train.drop(['Treatment'], axis=True)
    y_train = scores_train['Treatment']
    colormap = {
        0: '#ff0000',  # Red
        1: '#0000ff',  # Blue
    }

    colorlist = [colormap[c] for c in y_train]
    targets = ['Frozen', 'Lyophilized']
    ax = feature.plot(x=0, y=1, kind='scatter', s=50, alpha=0.7, c=colorlist, figsize=(6, 6))

    ax.set_xlabel('Scores on LV 1')
    ax.set_ylabel('Scores on LV 2')
    ax.set_title("Scores Plot")
    plt.legend(targets)
    plt.show()


# function to plot Screeplot of Principal components
def screeplot(pca):
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()


# function to generate biplot for top 2 principal components
def biplot(datafile):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 0', fontsize=15)
    ax.set_ylabel('Principal Component 1', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['Lyophilized', 'Frozen']
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = datafile['Treatment'] == target
        ax.scatter(datafile.loc[indicesToKeep, 0]
                   , datafile.loc[indicesToKeep, 1]
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


# function to train models, display accuracy and confusion matrix and return predicted values
def models(model, traindata, testdata):
    x_train = traindata.drop(['Treatment'], axis=True)
    y_train = traindata['Treatment']

    x_test = testdata.drop(['Treatment'], axis=True)
    y_test = testdata['Treatment']

    if model == 'svclin':
        classifier = SVC(kernel='linear')

    elif model == 'svcrbf':
        classifier = SVC(kernel='rbf')

    elif model == 'svcpoly':
        classifier = SVC(kernel='poly')

    elif model == 'rf':
        classifier = RandomForestClassifier()

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print('confusion matrix for ' + model + ':\n', confusion_matrix(y_test, y_pred))
    print('evaluation metrics for ' + model + ':\n', classification_report(y_test, y_pred))

    return y_pred


# function to plot surface plot for model trained using top 2 components
def surface_plot(model, datafile):
    global classifier
    cmap = 'nipy_spectral'
    label_encode(datafile, 'Treatment')
    ax = plt.gca()
    ax.scatter(datafile[0], datafile[1], c=datafile['Treatment'], cmap=cmap)

    x0_lim = ax.get_xlim()
    x1_lim = ax.get_ylim()

    x0, x1 = np.meshgrid(np.linspace(*x0_lim, 500), np.linspace(*x1_lim, 500))

    feature0, feature1 = x0.flatten(), x1.flatten()
    feature0, feature1 = feature0.reshape((len(feature0), 1)), feature1.reshape((len(feature1), 1))
    x_test = np.hstack((feature0, feature1))

    if model == 'svclin':
        classifier = SVC(kernel='linear')
        name = "SVM Linear"

    elif model == 'svcrbf':
        classifier = SVC(kernel='rbf')
        name = "SVM RBF"

    elif model == 'svcpoly':
        classifier = SVC(kernel='poly')
        name = "SVM polynomial"

    elif model == 'rf':
        classifier = RandomForestClassifier()
        name = "Random Forest"

    classifier.fit(datafile[[0, 1]], datafile['Treatment'])
    print('Training Accuracy = ', 100 * accuracy_score(datafile['Treatment'], classifier.predict(datafile[[0, 1]])))

    y_pred = classifier.predict(x_test)
    z = y_pred.reshape(x0.shape)

    n_classes = len(np.unique(y_pred))
    ax.contourf(x0, x1, z, levels=np.arange(n_classes + 1) - 0.5, alpha=0.5, cmap=cmap)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_title("Surface plot of " + name)
    plt.show()


# function to plot components based on MSE and MAE values
def plot_comp(component, metric, metrictype, DRT_type):
    metric_min = np.argmin(metric)
    with plt.style.context('ggplot'):
        plt.plot(component, np.array(metric), '-v', color='blue', mfc='blue')
        plt.plot(component[metric_min], np.array(metric)[metric_min], 'P', ms=10, mfc='red')
        plt.xlabel('Number of components')
        plt.ylabel(metrictype)
        plt.title(DRT_type)
        plt.xlim(left=-1)
    plt.show()


# function to train models on regression dataset and return predicted values.
def models_reg(model, Xpca, y):
    global classif
    if model == 'svclin':
        classif = SVR(kernel='linear')

    elif model == 'svcrbf':
        classif = SVR(kernel='rbf')

    elif model == 'svcpoly':
        classif = SVR(kernel='poly')

    elif model == 'lr':
        classif = LinearRegression()

    elif model == 'en':
        classif = ElasticNet()

    elif model == 'dtr':
        classif = DecisionTreeRegressor()

    classif.fit(Xpca, y)
    y_c = classif.predict(Xpca)
    y_cv = cross_val_predict(classif, Xpca, y, cv=10)

    return y_c, y_cv


# function to plot scatter plot for response variables with regression lines
def regression_plot(y, y_cv):
    score_cv = r2_score(y, y_cv)
    z = np.polyfit(y, y_cv, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y, c='red', edgecolors='k')
        ax.plot(z[1] + z[0] * y, y, c='blue', linewidth=1)
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): ' + str(score_cv))
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.show()


# function to apply PCA on regression dataset to predict response variables using optimal number of components.
# use these components to calculate R-squared values and plot regression line
def pcr(model, X, y, pc, plot_components = True):
    global Xstd
    mse = []
    mae = []
    component = np.arange(1, pc)

    for i in component:
        Xstd = StandardScaler().fit_transform(X)

        pca = PCA(n_components = i)
        Xpca = pca.fit_transform(Xstd)
        y_c, y_cv = models_reg(model, Xpca, y)

        mse.append(mean_squared_error(y, y_cv))
        mae.append(mean_absolute_error(y, y_cv))

    msemin = np.argmin(mse)
    maemin = np.argmin(mae)

    print("Suggested number of components w.r.t mse: ", msemin+1)
    print("Suggested number of components w.r.t mae: ", maemin+1)

    if plot_components is True:
        plot_comp(component, mse, 'MSE', 'PCR')
        plot_comp(component, mae, 'MAE', 'PCR')

    pca_opt = PCA(n_components = msemin + 1)
    Xpca = pca_opt.fit_transform(Xstd)

    y_c, y_cv = models_reg(model, Xpca, y)

    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    mae_c = mean_absolute_error(y, y_c)
    mae_cv = mean_absolute_error(y, y_cv)

    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    print('MAE calib: %5.3f' % mae_c)
    print('MAE CV: %5.3f' % mae_cv)

    regression_plot(y, y_cv)
    return

# function to apply PLS on regression dataset to predict response variables using optimal number of components.
# # use these components to calculate R-squared values and plot regression line
def plsr(X, y, n_comp, plot_components=True):
    mse = []
    mae = []
    component = np.arange(1, n_comp)

    for i in component:
        pls = PLSRegression(n_components = i)

        y_cv = cross_val_predict(pls, X, y, cv=10)

        mse.append(mean_squared_error(y, y_cv))
        mae.append(mean_absolute_error(y, y_cv))

    msemin = np.argmin(mse)
    maemin = np.argmin(mae)

    print("Suggested number of components w.r.t mse: ", msemin+1)
    print("Suggested number of components w.r.t mae: ", maemin+1)

    if plot_components is True:
        plot_comp(component, mse, 'MSE', 'PLSR')
        plot_comp(component, mae, 'MAE', 'PLSR')

    pls_opt = PLSRegression(n_components = msemin+1)

    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)

    y_cv = cross_val_predict(pls_opt, X, y, cv=10)

    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    mae_c = mean_absolute_error(y, y_c)
    mae_cv = mean_absolute_error(y, y_cv)

    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    print('MAE calib: %5.3f' % mae_c)
    print('MAE CV: %5.3f' % mae_cv)

    regression_plot(y, y_cv)
    return

# function to generate report of descriptive statistics for imported datasets
from dataprep.eda import create_report, plot
def report(datafile):
    report = create_report(datafile)
    report.show_browser()
    plot(datafile).show_browser()
    print(datafile.describe())
