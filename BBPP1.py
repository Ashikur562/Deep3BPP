{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zP0mgl4QkKMc"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sb"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install catboost"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l-zq1d5CkfU1",
        "outputId": "f726e5aa-1466-4b66-e649-3ab13ebd250f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting catboost\n",
            "  Downloading catboost-1.2-cp310-cp310-manylinux2014_x86_64.whl (98.6 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m98.6/98.6 MB\u001b[0m \u001b[31m6.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: graphviz in /usr/local/lib/python3.10/dist-packages (from catboost) (0.20.1)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from catboost) (3.7.1)\n",
            "Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.10/dist-packages (from catboost) (1.22.4)\n",
            "Requirement already satisfied: pandas>=0.24 in /usr/local/lib/python3.10/dist-packages (from catboost) (1.5.3)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from catboost) (1.10.1)\n",
            "Requirement already satisfied: plotly in /usr/local/lib/python3.10/dist-packages (from catboost) (5.13.1)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from catboost) (1.16.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24->catboost) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24->catboost) (2022.7.1)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (1.1.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (4.41.0)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (1.4.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (8.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (3.1.0)\n",
            "Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly->catboost) (8.2.2)\n",
            "Installing collected packages: catboost\n",
            "Successfully installed catboost-1.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score\n",
        "from sklearn.metrics import roc_auc_score\n",
        "from sklearn.metrics import precision_recall_curve\n",
        "from sklearn.metrics import matthews_corrcoef\n",
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "mx_AEPx3kh98"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from sklearn.ensemble import AdaBoostClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from catboost import CatBoostClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.ensemble import StackingClassifier\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.model_selection import cross_val_score, cross_val_predict"
      ],
      "metadata": {
        "id": "0eLswzzykkW9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "estimator = [('RF', RandomForestClassifier(n_estimators = 215, max_depth = 9)), ('XGB', XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163)),\n",
        "             ('Cat', CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025)), ('LGBM', LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96)), ('DT', DecisionTreeClassifier(max_depth = 5)),\n",
        "             ('ETC', ExtraTreesClassifier(n_estimators = 646, max_depth = 9)), ('GBC', GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014)),\n",
        "             ('ADB', AdaBoostClassifier(n_estimators = 800, learning_rate = 0.5050920867773578, random_state = 66)), ('KNN', KNeighborsClassifier(n_neighbors=1))]\n",
        "Stacking = StackingClassifier( estimators=estimator, final_estimator= RandomForestClassifier(n_estimators = 215, max_depth = 9))"
      ],
      "metadata": {
        "id": "kCStcZIeknIt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Word2Vec**"
      ],
      "metadata": {
        "id": "Vu2295Ook-6J"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "wi5q6yVZXDIs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Word2Vec/Wor2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "q1WsXu8ZlFMd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Word2Vec-CV).csv\")"
      ],
      "metadata": {
        "id": "wRZVH8v-lJ2H",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9df9bda3-e2b6-4487-fe48-e58dbe3c2ae4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=9, n_estimato...  0.759690 -0.020530   \n",
            "1  XGBClassifier(base_score=0.1077705548489194, b...  0.750078  0.182396   \n",
            "2  <catboost.core.CatBoostClassifier object at 0x...  0.748837  0.165467   \n",
            "3  LGBMClassifier(learning_rate=0.211371201238646...  0.760310  0.232834   \n",
            "4                DecisionTreeClassifier(max_depth=5)  0.777364  0.081941   \n",
            "5  ExtraTreesClassifier(max_depth=9, n_estimators...  0.753178 -0.018996   \n",
            "6  GradientBoostingClassifier(learning_rate=0.132...  0.753798  0.201246   \n",
            "7                KNeighborsClassifier(n_neighbors=1)  0.797829  0.574176   \n",
            "8  StackingClassifier(estimators=[('RF',\\n       ...  0.930233  0.805533   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0 -0.016631   0.167513  0.051242  0.078478     0.936459     0.051242  \n",
            "1  0.181852   0.357394  0.315217  0.334983     0.858582     0.315217  \n",
            "2  0.164549   0.346863  0.291925  0.317032     0.862844     0.291925  \n",
            "3  0.232659   0.393388  0.369565  0.381105     0.857807     0.369565  \n",
            "4  0.067690   0.323810  0.105590  0.159251     0.944983     0.105590  \n",
            "5 -0.016157   0.172414  0.062112  0.091324     0.925610     0.062112  \n",
            "6  0.200854   0.371134  0.335404  0.352365     0.858194     0.335404  \n",
            "7  0.523513   0.496683  0.930124  0.647568     0.764820     0.930124  \n",
            "8  0.798728   0.765526  0.937888  0.842987     0.928322     0.937888  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "RVHUhx-UW9bs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Word2Vec/Wor2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "_Ym1S0vbXASX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "xtrain, ytrain = ada.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "8EUzS8E5XONO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Word2Vec-CV(ADASYN)).csv\")"
      ],
      "metadata": {
        "id": "jbOEp_e-XQ6U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTEN**"
      ],
      "metadata": {
        "id": "h_IlsR5vXR04"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Word2Vec/Wor2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "szeIZ4J_XTo0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import SMOTEN\n",
        "smn = SMOTEN()\n",
        "xtrain, ytrain = smn.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "9gVB3DqFXV_k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Word2Vec-CV(SMOTEN)).csv\")"
      ],
      "metadata": {
        "id": "0Nxb7tyVXdPQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "gBEAgugLXhJU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Word2Vec/Wor2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "NtCxDHk9Xj73"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "xtrain, ytrain = smt.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "4uG3OuOAXnwN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Word2Vec-CV(SMOTETomek)).csv\")"
      ],
      "metadata": {
        "id": "aQ9qmfMQXpwk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Nearmiss**"
      ],
      "metadata": {
        "id": "Bc6NGwfKXu13"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Word2Vec/Wor2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "hiFv15lOXxWT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "xtrain, ytrain = nm.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "l63oo-JcXz1L"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Word2Vec-CV(NearMiss)).csv\")"
      ],
      "metadata": {
        "id": "CNWZIbRlX2dl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**TomekLinks**"
      ],
      "metadata": {
        "id": "RzM4LGw9X7Ar"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Word2Vec/Wor2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "bUJilulWX9D9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import TomekLinks\n",
        "tl = TomekLinks()\n",
        "xtrain, ytrain = tl.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "n-0GRzHUX_ZL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Word2Vec-CV(TomekLinks)).csv\")"
      ],
      "metadata": {
        "id": "BDod43tsYCh9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Doc2Vec**"
      ],
      "metadata": {
        "id": "0Xgtj9E0lNfe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "PZa4ouOiYdXN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Doc2Vec/Doc2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "bIF6Lj3-lS8F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Doc2Vec-CV).csv\")"
      ],
      "metadata": {
        "id": "KzYDOXuAlU70"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "XOW8txKDYmD9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Doc2Vec/Doc2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "VI2ZJrASYoV8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "xtrain, ytrain = ada.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "cGPuWVOoYrbs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Doc2Vec-CV(ADASYN)).csv\")"
      ],
      "metadata": {
        "id": "HtI49b50YuWE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTEN**"
      ],
      "metadata": {
        "id": "ke0YvacjY0jd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Doc2Vec/Doc2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "dHije_laY2dj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import SMOTEN\n",
        "smn = SMOTEN()\n",
        "xtrain, ytrain = smn.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "fYHMYQ0iY4wj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Doc2Vec-CV(SMOTEN)).csv\")"
      ],
      "metadata": {
        "id": "yK9wHwnSY8eL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "TgnkILXFZCEF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Doc2Vec/Doc2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "i8LXN_vvZEpz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "xtrain, ytrain = smt.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "5mCbTEJ3ZHcL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Doc2Vec-CV(SMOTETomek)).csv\")"
      ],
      "metadata": {
        "id": "Czwye0pvZMRD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "vxROQHtrZQEa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Doc2Vec/Doc2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "6LMhkVWBZSR8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "xtrain, ytrain = nm.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "g5J4iiIxZUWT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Doc2Vec-CV(NearMiss)).csv\")"
      ],
      "metadata": {
        "id": "UhBtj72qZXeM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**TomekLinks**"
      ],
      "metadata": {
        "id": "U56b6akRZbau"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df_tr = pd.read_csv('/content/drive/MyDrive/Bioinformatics/BBB PP/Encodeing Data/Doc2Vec/Doc2Vec-TR.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "xtrain = df_tr[columns]\n",
        "ytrain = df_tr[target]"
      ],
      "metadata": {
        "id": "mBDAZhaOZhWJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import TomekLinks\n",
        "tl = TomekLinks()\n",
        "xtrain, ytrain = tl.fit_resample(xtrain, ytrain)"
      ],
      "metadata": {
        "id": "dLhfDFeEZlGl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 215, max_depth = 9),\n",
        "          XGBClassifier(n_estimators = 209,max_depth = 9, base_score = 0.1077705548489194, learning_rate = 0.066495650542163),\n",
        "          CatBoostClassifier(depth= 8, iterations = 24, learning_rate = 0.3309357576147025),\n",
        "          LGBMClassifier(learning_rate = 0.21137120123864672,max_depth = 9,random_state = 96),\n",
        "          DecisionTreeClassifier(max_depth = 5),\n",
        "          ExtraTreesClassifier(n_estimators = 646, max_depth = 9),\n",
        "          GradientBoostingClassifier(max_depth = 8, n_estimators = 911, learning_rate = 0.13235577168387014),\n",
        "          KNeighborsClassifier(n_neighbors=1),\n",
        "          Stacking]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"/content/drive/MyDrive/Bioinformatics/BBB PP/Result/total_Metics(Doc2Vec-CV(TomekLinks)).csv\")"
      ],
      "metadata": {
        "id": "NdeQgz1BZndV"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
