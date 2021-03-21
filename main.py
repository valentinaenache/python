import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as discrim
import sklearn.metrics as metrics
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.tree as tree
import grafice

# Citire date si creare dataframe prin biblioteca Pandas
tabel = pd.read_csv("Date2019.csv", index_col=0)
pd.set_option('display.max_columns', None)
#print(tabel)

# Inlocuirea valorilor lipsa cu media
tabel.replace(" ", np.nan, inplace=True)
date_lipsa = tabel.isnull()
#print(date_lipsa)

# for column in date_lipsa.columns.values.tolist():
#     print(column)
#     print(date_lipsa[column].value_counts())
#     print("")

# Date lipsa: SalariuMediu, ProcentSalariuMinim, ExportImportRatio
medie_sal_mediu = tabel["SalariuMediu"].astype("float").mean(axis=0)
print("Media salariului mediu este:", medie_sal_mediu)
tabel["SalariuMediu"].replace(np.nan, medie_sal_mediu, inplace=True)

medie_procent_sal_minim = tabel["ProcentSalariuMinim"].astype("float").mean(axis=0)
print("Media procentului angajatilor platiti cu salariul minim pe economie este:", medie_procent_sal_minim)
tabel["ProcentSalariuMinim"].replace(np.nan, medie_procent_sal_minim, inplace=True)

medie_expimp_ratio = tabel["ExportImportRatio"].astype("float").mean(axis=0)
print("Media raportului dintre export si import este:", medie_expimp_ratio)
tabel["ExportImportRatio"].replace(np.nan, medie_expimp_ratio, inplace=True)

# Statistica descriptiva
indicatori_numerici = tabel.describe()
#print(indicatori_numerici)
indicatori_numerici.to_csv("Indicatori.csv")
print(tabel.describe())

# Aplicare model de clasificare LDA
variabile = list(tabel)
nr_variabile = len(variabile)
variabile_predictor = variabile[:(nr_variabile - 1)]
variabila_tinta = variabile[nr_variabile - 1]
print("Variabila tinta este:", variabila_tinta)
print("Variabilele predictoare sunt:", variabile_predictor)

x = tabel[variabile_predictor].values
y = tabel[variabila_tinta].values
# print(type(x))
# print(type(y))


# Construire model si preluare etichete regiuni
model_lda= discrim.LinearDiscriminantAnalysis()
model_lda.fit(x, np.ravel(y))
regiuni = model_lda.classes_
print("Regiuni:", regiuni)

# Preluare rezultate si aplicare model
# Calcul scoruri discriminante
z = model_lda.transform(x)
n, q = z.shape
etichete_z = ["z" + str(i) for i in range(1, q + 1)]
nume_instante = list(tabel.index)

# Tabel scoruri
t_z = pd.DataFrame(z, nume_instante, etichete_z)
t_z.to_csv("z.csv")

# Calculare centru de grupe
g = model_lda.means_
zg = model_lda.transform(g)
if q > 1:
    grafice.biplot(z, zg, y, regiuni)
for i in range(q):
    grafice.distributie(z,i,y,regiuni)

# Clasificare in setul de invatare
clasificare_b = model_lda.predict(x)
tabel_clasificare_b = pd.DataFrame(
        data={
         "Regiunea": y,
         "Predictie": clasificare_b
     }, index=nume_instante
)
tabel_clasificare_b.to_csv("clasif_b.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err = tabel_clasificare_b[y != clasificare_b]
#tabel_clasificare_err.to_csv("clasif_eronata.csv")

# Calcul matrice clasificari eronate si aplicare pe setul de invatare
mat_conf = metrics.confusion_matrix(y, clasificare_b)
t_mat_conf = pd.DataFrame(mat_conf, regiuni, regiuni)
t_mat_conf["Acuratete"] = np.diagonal(mat_conf) * 100 / np.sum(mat_conf, axis=1)
print(t_mat_conf)
#t_mat_conf.to_csv("mat_conf.csv")
acuratete_globala = sum(np.diagonal(mat_conf)) * 100 / n
print("Acuratete globala:", sum(np.diagonal(mat_conf)) * 100 / n)


# Aplicarea modelului pe test (date 2014)
set_testare = pd.read_csv("Date2014.csv", index_col=0)
x_testare = set_testare[variabile_predictor].values
predictie = model_lda.predict(x_testare)
set_testare["Predictie_lda"] = predictie

# Clasificare in setul de testare
clasificare_test= pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie" : predictie
    }, index=nume_instante
)
clasificare_test.to_csv("clasif_test.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_test = clasificare_test[y != predictie]
#tabel_clasificare_err_test.to_csv("clasif_eronata_test.csv")

# Calcul matrice clasificari eronate si aplicare pe setul de testare
mat_conf_test = metrics.confusion_matrix(y, predictie)
t_mat_conf_test = pd.DataFrame(mat_conf_test, regiuni,regiuni)
t_mat_conf_test["Acuratete"] = np.diagonal(mat_conf_test) * 100 / np.sum(mat_conf_test, axis=1)
print(t_mat_conf_test)
#t_mat_conf_test.to_csv("mat_conf_test.csv")
acuratete_test = sum(np.diagonal(mat_conf_test)) * 100 / n
print("Acuratete globala pentru setul de testare:", sum(np.diagonal(mat_conf_test)) * 100 / n)
#
# Construire model bayesian
model_bayes = nb.GaussianNB()
model_bayes.fit(x, y)

# Clasificare in setul de invatare
clasificare_b_bayes = model_bayes.predict(x)
tabel_clasificare_b_bayes = pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie": clasificare_b_bayes
    },index=nume_instante
)
tabel_clasificare_b_bayes.to_csv("clasif_b_bayes.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_bayes = tabel_clasificare_b_bayes[y != clasificare_b_bayes]
#tabel_clasificare_err_bayes.to_csv("clasif_eronata_bayes.csv")

# Calcul matrice clasificari eronate si aplicare pe setul de invatare
mat_conf_bayes = metrics.confusion_matrix(y, clasificare_b_bayes)
t_mat_conf_bayes = pd.DataFrame(mat_conf_bayes, regiuni, regiuni)
t_mat_conf_bayes["Acuratete"] = np.diagonal(mat_conf_bayes) * 100 / np.sum(mat_conf_bayes, axis=1)
print(t_mat_conf_bayes)
#t_mat_conf_bayes.to_csv("mat_conf_bayes.csv")
acuratete_globala_bayes = sum(np.diagonal(mat_conf_bayes)) * 100 / n
print("Acuratete globala model bayesian:", sum(np.diagonal(mat_conf_bayes)) * 100 / n)


# Aplicarea modelului pe test (date 2014)
predictie_bayes = model_bayes.predict(x_testare)
set_testare["Predictie_Bayes"] = predictie_bayes

# Clasificare in setul de testare
clasificare_test_bayes= pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie" : predictie_bayes
    }, index=nume_instante
)
clasificare_test_bayes.to_csv("clasif_test_bayes.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_test_bayes = clasificare_test_bayes[y != predictie_bayes]
# tabel_clasificare_err_test_bayes.to_csv("clasif_eronata_test_bayes.csv")

# Calcul matrice clasificari eronate si aplicare pe setul de testare
mat_conf_test_bayes = metrics.confusion_matrix(y, predictie_bayes)
t_mat_conf_test_bayes = pd.DataFrame(mat_conf_test_bayes, regiuni,regiuni)
t_mat_conf_test_bayes["Acuratete"] = np.diagonal(mat_conf_test_bayes) * 100 / np.sum(mat_conf_test_bayes, axis=1)
print(t_mat_conf_test_bayes)
#t_mat_conf_test_bayes.to_csv("mat_conf_test_bayes.csv")
acuratete_test_bayes = sum(np.diagonal(mat_conf_test_bayes)) * 100 / n
print("Acuratete globala model bayesian pentru setul de testare:", sum(np.diagonal(mat_conf_test_bayes)) * 100 / n)


# Constuire model SVM
model_svm = svm.SVC()
model_svm.fit(x, y)

# Clasificare in setul de invatare
clasificare_b_svm = model_svm.predict(x)
tabel_clasificare_b_svm = pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie": clasificare_b_svm
    },index=nume_instante
)
tabel_clasificare_b_svm.to_csv("clasif_b_svm.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_svm = tabel_clasificare_b_svm[y != clasificare_b_svm]
# tabel_clasificare_err_svm.to_csv("clasif_eronata_svm.csv")

# Calcul matrice clasificari eronate si aplicare pe setul de invatare
mat_conf_svm = metrics.confusion_matrix(y, clasificare_b_svm)
t_mat_conf_svm = pd.DataFrame(mat_conf_svm, regiuni, regiuni)
t_mat_conf_svm["Acuratete"] = np.diagonal(mat_conf_svm) * 100 / np.sum(mat_conf_svm, axis=1)
print(t_mat_conf_svm)
#t_mat_conf_svm.to_csv("mat_conf_svm.csv")
acuratete_globala_svm = sum(np.diagonal(mat_conf_svm)) * 100 / n
print("Acuratete globala model svm:", sum(np.diagonal(mat_conf_svm)) * 100 / n)

# Aplicarea modelului pe test (date 2014)
predictie_svm = model_svm.predict(x_testare)
set_testare["Predictie_SVM"] = predictie_svm

# Clasificare in setul de testare
clasificare_test_svm= pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie" : predictie_svm
    }, index=nume_instante
)
clasificare_test_svm.to_csv("clasif_test_svm.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_test_svm = clasificare_test_svm[ y!= predictie_svm]
# tabel_clasificare_err_test_svm.to_csv("clasif_eronata_test_bayes.csv")

#Calcul matrice clasificari eronate si aplicare pe setul de testare
mat_conf_test_svm = metrics.confusion_matrix(y, predictie_svm)
t_mat_conf_test_svm = pd.DataFrame(mat_conf_test_svm, regiuni, regiuni)
t_mat_conf_test_svm["Acuratete"] = np.diagonal(mat_conf_test_svm) * 100 / np.sum(mat_conf_test_svm, axis=1)
print(t_mat_conf_test_svm)
#t_mat_conf_test_svm.to_csv("mat_conf_test_svm.csv")
acuratete_test_svm = sum(np.diagonal(mat_conf_test_svm)) * 100 / n
print("Acuratete globala model SVM pentru setul de testare:",sum(np.diagonal(mat_conf_test_svm)) * 100 / n)


# Constuire model arbore de decizie
model_arbore = tree.DecisionTreeClassifier()
model_arbore.fit(x, y)

# Aplicare pe setul de invatare
clasificare_b_arbore = model_arbore.predict(x)
tabel_clasificare_b_arbore = pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie" : clasificare_b_arbore
    }, index=nume_instante
)
tabel_clasificare_b_arbore.to_csv("clasif_b_arbore.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_arbore = tabel_clasificare_b_arbore[y != clasificare_b_arbore]
# tabel_clasificare_err_arbore.to_csv("clasif_eronata_arbore.csv")

# Calcul matrice clasificari eronate
mat_conf_arbore = metrics.confusion_matrix(y, clasificare_b_arbore)
t_mat_conf_arbore = pd.DataFrame(mat_conf_arbore, regiuni, regiuni)
t_mat_conf_arbore["Acuratete"] = np.diagonal(mat_conf_arbore) * 100 / np.sum(mat_conf_arbore, axis=1)
print(t_mat_conf_arbore)
#t_mat_conf_arbore.to_csv("mat_conf_arbore.csv")
acuratete_globala_arbore = sum(np.diagonal(mat_conf_arbore)) * 100 / n
print("Acuratete globala model arbore:", sum(np.diagonal(mat_conf_arbore)) * 100 / n)

# Aplicarea modelului pe test (date 2014)
predictie_arbore = model_arbore.predict(x_testare)
set_testare["Predictie_arbore"] = predictie_arbore

# Clasificare in setul de testare
clasificare_test_arbore= pd.DataFrame(
    data = {
        "Regiunea" : y,
        "Predictie" : predictie_arbore
    }, index=nume_instante
)
clasificare_test_arbore.to_csv("clasif_test_arbore.csv")

# Izolarea instantelor clasificate eronat
tabel_clasificare_err_test_arbore = clasificare_test_arbore[y != predictie_arbore]
# tabel_clasificare_err_test_arbore.to_csv("clasif_eronata_test_arbore.csv")

# Clasificare_b_arbore = model_arbore.predict(x)
mat_conf_test_arbore = metrics.confusion_matrix(y, predictie_arbore)
t_mat_conf_test_arbore = pd.DataFrame(mat_conf_test_arbore, regiuni, regiuni)
t_mat_conf_test_arbore["Acuratete"] = np.diagonal(mat_conf_test_arbore) * 100 / np.sum(mat_conf_test_arbore, axis=1)
print(t_mat_conf_test_arbore)
#t_mat_conf_test_arbore.to_csv("mat_conf_test_arbore.csv")
acuratete_test_arbore = sum(np.diagonal(mat_conf_test_arbore)) * 100 /n
print("Acuratete globala model arbore pentru setul de testare:", sum(np.diagonal(mat_conf_test_arbore)) * 100 /n)

# Salvarea rezultatelor
set_testare.to_csv("Predictie.csv")

acuratete_calculata = np.array([acuratete_test, acuratete_test_bayes, acuratete_test_svm, acuratete_test_arbore])
predictii_set_testare = pd.Series(acuratete_calculata, index = ['LDA', 'Bayes', 'SVM', 'Arbore'])
# predictii_grupate_test = predictii_set_testare.groupby("Acuratete")
print(predictii_set_testare)

acuratete_globala = [acuratete_globala, acuratete_globala_bayes, acuratete_globala_svm, acuratete_globala_arbore]
predictii_set_invatare = pd.DataFrame(
    data = {
        "Tipul Clasificarii Supervizate Set Invatare": ['LDA', 'Bayes', 'SVM', 'Arbore'],
        "Acuratete" : acuratete_globala
    }
)
predictii_grupate = predictii_set_invatare.groupby(by="Acuratete")
print(predictii_grupate.first())

grafice.show()