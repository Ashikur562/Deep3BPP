{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "zPpqYcYee-Bu",
        "qbL9jyMhe6o2",
        "zn0iRJUc6DVR",
        "5_ShZ0_OAHLu",
        "AbE79UmWNZlG",
        "nqcqt24FT2Hs",
        "u976iNwoZMBT",
        "fZ_RE4JEamAc"
      ]
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
        "id": "7Zwpm0OlXJJD"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sb\n",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score\n",
        "from sklearn.metrics import matthews_corrcoef\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from sklearn.model_selection import cross_val_score, cross_val_predict"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Conv1D, MaxPool1D, Flatten,LSTM"
      ],
      "metadata": {
        "id": "1FC0yWG8XW5y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CNN+LSTM(APAAC)**"
      ],
      "metadata": {
        "id": "zPpqYcYee-Bu"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "QkHO2TEsmjyE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "tk16hRv2XYjX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "3plPdWo9XdtA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "r-U6wIvzXfgH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "9T_ad9FIXhev"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "teFjgTQXXjaX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "8E2b-4a9XlKw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "jQLxzE_yXnLH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "KabivOn4Xqe6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "6v6wx7B1XsLn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "OCvipUByX0Q3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "5gOFGdkCX0zK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VmSP0fKCX3KY",
        "outputId": "01437d1f-9c12-4667-d581-067a41912139"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 5s 70ms/step - loss: 0.5155 - accuracy: 0.7789\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 3s 67ms/step - loss: 0.4805 - accuracy: 0.7974\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.4690 - accuracy: 0.7974\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.4380 - accuracy: 0.7974\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.4143 - accuracy: 0.8095\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.3930 - accuracy: 0.8172\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.3792 - accuracy: 0.8280\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.3672 - accuracy: 0.8431\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.3613 - accuracy: 0.8482\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.3378 - accuracy: 0.8598\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.3263 - accuracy: 0.8687\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.3041 - accuracy: 0.8838\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.2906 - accuracy: 0.8842\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.2797 - accuracy: 0.8916\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.2611 - accuracy: 0.8993\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.2660 - accuracy: 0.8993\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 3s 69ms/step - loss: 0.2309 - accuracy: 0.9082\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.2136 - accuracy: 0.9160\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.1794 - accuracy: 0.9233\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.2347 - accuracy: 0.9094\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.1635 - accuracy: 0.9299\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.1523 - accuracy: 0.9330\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.1483 - accuracy: 0.9400\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.1149 - accuracy: 0.9531\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0929 - accuracy: 0.9620\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.0727 - accuracy: 0.9713\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 3s 67ms/step - loss: 0.0762 - accuracy: 0.9679\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.0629 - accuracy: 0.9752\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0926 - accuracy: 0.9648\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0889 - accuracy: 0.9682\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0524 - accuracy: 0.9822\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.0619 - accuracy: 0.9775\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0349 - accuracy: 0.9876\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.0147 - accuracy: 0.9957\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 3s 74ms/step - loss: 0.0127 - accuracy: 0.9961\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.0125 - accuracy: 0.9965\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0082 - accuracy: 0.9969\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0050 - accuracy: 0.9977\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0080 - accuracy: 0.9973\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0246 - accuracy: 0.9907\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0335 - accuracy: 0.9888\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0531 - accuracy: 0.9779\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.0191 - accuracy: 0.9938\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 3s 78ms/step - loss: 0.0214 - accuracy: 0.9919\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0161 - accuracy: 0.9957\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0062 - accuracy: 0.9981\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.0066 - accuracy: 0.9981\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.0094 - accuracy: 0.9965\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0046 - accuracy: 0.9985\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0038 - accuracy: 0.9985\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.0038 - accuracy: 0.9981\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.0060 - accuracy: 0.9981\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0038 - accuracy: 0.9985\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 3s 76ms/step - loss: 0.0033 - accuracy: 0.9985\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.0039 - accuracy: 0.9985\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0709 - accuracy: 0.9768\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.0286 - accuracy: 0.9911\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 3s 78ms/step - loss: 0.0260 - accuracy: 0.9926\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.0209 - accuracy: 0.9938\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0051 - accuracy: 0.9981\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0067 - accuracy: 0.9977\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.0156 - accuracy: 0.9934\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.0077 - accuracy: 0.9977\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.0043 - accuracy: 0.9977\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.0038 - accuracy: 0.9981\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 3s 74ms/step - loss: 0.0066 - accuracy: 0.9981\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.0035 - accuracy: 0.9985\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0031 - accuracy: 0.9988\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0030 - accuracy: 0.9985\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0025 - accuracy: 0.9992\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 0.0020 - accuracy: 0.9992\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.0084 - accuracy: 0.9973\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.0043 - accuracy: 0.9985\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.0034 - accuracy: 0.9985\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0027 - accuracy: 0.9988\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7ed228227490>"
            ]
          },
          "metadata": {},
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f8PIYcFPX41v",
        "outputId": "43077a86-b7c6-46e3-dee7-a878d778f1b1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 1s 11ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h3u-oCWaX76n",
        "outputId": "c889c2f2-fb33-42df-ddd3-578ff3edc416"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9798449612403101,\n",
              " 1.0,\n",
              " 0.9037037037037037,\n",
              " 0.9368718238283456,\n",
              " 0.9387442172032185)"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J2l0oN5KX-4Q",
        "outputId": "7ef41288-638e-4099-9247-8bedff1f357a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9751434034416826"
            ]
          },
          "metadata": {},
          "execution_count": 63
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "r6BaYGZfgRNq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "KiCvk2IAgS0r"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "QKKDbR7_gWpB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "rt-5kpJCgZ0U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "j-ZIV7xlgdbB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "4ltNTaYogfvL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "# cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "feYtRXi8ghq5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "SgJh3dICgjoQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "sI8-qkSjglt_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "OdaNniNcgn6J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "EKRfmvmWgqEZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "ROvKCisAgr6E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tih12KOOguzz",
        "outputId": "df9f2b7d-9080-4354-8df5-1263b5a407f2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 6s 91ms/step - loss: 0.5024 - accuracy: 0.7967\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 3s 93ms/step - loss: 0.4646 - accuracy: 0.8127\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.4549 - accuracy: 0.8127\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 3s 82ms/step - loss: 0.4535 - accuracy: 0.8127\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 3s 86ms/step - loss: 0.4136 - accuracy: 0.8175\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 3s 95ms/step - loss: 0.3766 - accuracy: 0.8299\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 3s 80ms/step - loss: 0.3479 - accuracy: 0.8388\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 0.3526 - accuracy: 0.8406\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 3s 89ms/step - loss: 0.3287 - accuracy: 0.8543\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 3s 90ms/step - loss: 0.3472 - accuracy: 0.8565\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.3102 - accuracy: 0.8702\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 3s 84ms/step - loss: 0.2953 - accuracy: 0.8840\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 3s 96ms/step - loss: 0.2757 - accuracy: 0.8933\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 3s 82ms/step - loss: 0.2631 - accuracy: 0.9017\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 0.2505 - accuracy: 0.9039\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.2378 - accuracy: 0.9035\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 4s 98ms/step - loss: 0.2108 - accuracy: 0.9194\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.2186 - accuracy: 0.9172\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 3s 80ms/step - loss: 0.1993 - accuracy: 0.9172\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.1805 - accuracy: 0.9274\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.1763 - accuracy: 0.9327\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.1824 - accuracy: 0.9265\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.1520 - accuracy: 0.9438\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.1367 - accuracy: 0.9455\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.1447 - accuracy: 0.9429\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.1633 - accuracy: 0.9331\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 0.1171 - accuracy: 0.9522\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0846 - accuracy: 0.9686\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.1319 - accuracy: 0.9570\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.1168 - accuracy: 0.9548\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0840 - accuracy: 0.9699\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 0.0575 - accuracy: 0.9761\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0561 - accuracy: 0.9756\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0409 - accuracy: 0.9845\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0942 - accuracy: 0.9632\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0643 - accuracy: 0.9787\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0406 - accuracy: 0.9827\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0440 - accuracy: 0.9849\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0247 - accuracy: 0.9916\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0307 - accuracy: 0.9911\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0888 - accuracy: 0.9730\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 0.0546 - accuracy: 0.9810\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0445 - accuracy: 0.9845\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0287 - accuracy: 0.9903\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0304 - accuracy: 0.9911\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0124 - accuracy: 0.9951\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0259 - accuracy: 0.9916\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.0680 - accuracy: 0.9774\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0202 - accuracy: 0.9925\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0440 - accuracy: 0.9863\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0543 - accuracy: 0.9823\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0306 - accuracy: 0.9907\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 2s 67ms/step - loss: 0.0103 - accuracy: 0.9965\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0094 - accuracy: 0.9965\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0055 - accuracy: 0.9982\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0038 - accuracy: 0.9987\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 2s 67ms/step - loss: 0.0030 - accuracy: 0.9991\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 3s 69ms/step - loss: 0.0025 - accuracy: 0.9991\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0023 - accuracy: 0.9991\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0022 - accuracy: 0.9991\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0022 - accuracy: 0.9991\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.0021 - accuracy: 0.9991\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 0.0020 - accuracy: 0.9991\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0021 - accuracy: 0.9991\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0033 - accuracy: 0.9982\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0032 - accuracy: 0.9982\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0375 - accuracy: 0.9858\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 3s 78ms/step - loss: 0.0612 - accuracy: 0.9827\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0393 - accuracy: 0.9841\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0123 - accuracy: 0.9965\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0056 - accuracy: 0.9982\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.0047 - accuracy: 0.9991\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 3s 78ms/step - loss: 0.0356 - accuracy: 0.9885\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0299 - accuracy: 0.9911\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0245 - accuracy: 0.9916\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x795aa04ee3b0>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FKalPa_3gxXY",
        "outputId": "0ef5801b-4e94-4756-db3e-0deae9df7b13"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 8ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kul_JPe-g0xS",
        "outputId": "f4fbda4d-ba5e-47c5-f030-f098b0eae39d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.936016511867905,\n",
              " 0.8076923076923077,\n",
              " 0.9459459459459459,\n",
              " 0.8713692946058091,\n",
              " 0.8291384400116035,\n",
              " 0.8337225997100486)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NqLFqn0og5OA",
        "outputId": "0a26dcc0-bde4-496f-8bcd-c599e26a0eec"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9330655957161981"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "bDRwnlcpmnw1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "xOtZg5tBmpx-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "GjoGFedFms-s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "6GSZxmxQmwO2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "CCVers8xmy9D"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "ghlTH_lTm2uT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "R6OGlUkwm6Fj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "6D08Iya0m9PU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "BBsWmPHOnBC0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "uTJLjMEbnDdT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "mhC5ESyHnGF0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "edTPrR95nJX8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "SnWZAN2knMSm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4wRgvOa7nPGr",
        "outputId": "42ce00b7-0e42-40ac-b607-4dcb1c96306a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "64/64 [==============================] - 8s 79ms/step - loss: 0.6647 - accuracy: 0.6034\n",
            "Epoch 2/75\n",
            "64/64 [==============================] - 4s 70ms/step - loss: 0.6314 - accuracy: 0.6481\n",
            "Epoch 3/75\n",
            "64/64 [==============================] - 5s 82ms/step - loss: 0.5734 - accuracy: 0.7078\n",
            "Epoch 4/75\n",
            "64/64 [==============================] - 5s 70ms/step - loss: 0.5298 - accuracy: 0.7365\n",
            "Epoch 5/75\n",
            "64/64 [==============================] - 4s 69ms/step - loss: 0.4453 - accuracy: 0.7895\n",
            "Epoch 6/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.3859 - accuracy: 0.8227\n",
            "Epoch 7/75\n",
            "64/64 [==============================] - 4s 68ms/step - loss: 0.3064 - accuracy: 0.8635\n",
            "Epoch 8/75\n",
            "64/64 [==============================] - 5s 76ms/step - loss: 0.2535 - accuracy: 0.8922\n",
            "Epoch 9/75\n",
            "64/64 [==============================] - 5s 71ms/step - loss: 0.2067 - accuracy: 0.9167\n",
            "Epoch 10/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.1685 - accuracy: 0.9408\n",
            "Epoch 11/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.1386 - accuracy: 0.9521\n",
            "Epoch 12/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.1524 - accuracy: 0.9450\n",
            "Epoch 13/75\n",
            "64/64 [==============================] - 5s 83ms/step - loss: 0.1167 - accuracy: 0.9605\n",
            "Epoch 14/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0711 - accuracy: 0.9774\n",
            "Epoch 15/75\n",
            "64/64 [==============================] - 4s 68ms/step - loss: 0.0485 - accuracy: 0.9860\n",
            "Epoch 16/75\n",
            "64/64 [==============================] - 6s 87ms/step - loss: 0.0336 - accuracy: 0.9897\n",
            "Epoch 17/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0773 - accuracy: 0.9715\n",
            "Epoch 18/75\n",
            "64/64 [==============================] - 5s 82ms/step - loss: 0.1062 - accuracy: 0.9610\n",
            "Epoch 19/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0275 - accuracy: 0.9904\n",
            "Epoch 20/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0231 - accuracy: 0.9921\n",
            "Epoch 21/75\n",
            "64/64 [==============================] - 6s 92ms/step - loss: 0.0148 - accuracy: 0.9971\n",
            "Epoch 22/75\n",
            "64/64 [==============================] - 6s 86ms/step - loss: 0.0354 - accuracy: 0.9875\n",
            "Epoch 23/75\n",
            "64/64 [==============================] - 5s 84ms/step - loss: 0.0308 - accuracy: 0.9902\n",
            "Epoch 24/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0453 - accuracy: 0.9880\n",
            "Epoch 25/75\n",
            "64/64 [==============================] - 6s 87ms/step - loss: 0.0183 - accuracy: 0.9936\n",
            "Epoch 26/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0073 - accuracy: 0.9985\n",
            "Epoch 27/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0104 - accuracy: 0.9980\n",
            "Epoch 28/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0049 - accuracy: 0.9988\n",
            "Epoch 29/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0108 - accuracy: 0.9975\n",
            "Epoch 30/75\n",
            "64/64 [==============================] - 5s 81ms/step - loss: 0.0101 - accuracy: 0.9968\n",
            "Epoch 31/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 32/75\n",
            "64/64 [==============================] - 6s 87ms/step - loss: 0.0041 - accuracy: 0.9990\n",
            "Epoch 33/75\n",
            "64/64 [==============================] - 5s 76ms/step - loss: 0.0036 - accuracy: 0.9990\n",
            "Epoch 34/75\n",
            "64/64 [==============================] - 4s 68ms/step - loss: 0.0032 - accuracy: 0.9993\n",
            "Epoch 35/75\n",
            "64/64 [==============================] - 5s 84ms/step - loss: 0.0031 - accuracy: 0.9993\n",
            "Epoch 36/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0031 - accuracy: 0.9993\n",
            "Epoch 37/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.0329 - accuracy: 0.9885\n",
            "Epoch 38/75\n",
            "64/64 [==============================] - 5s 71ms/step - loss: 0.0597 - accuracy: 0.9811\n",
            "Epoch 39/75\n",
            "64/64 [==============================] - 5s 71ms/step - loss: 0.0123 - accuracy: 0.9961\n",
            "Epoch 40/75\n",
            "64/64 [==============================] - 5s 84ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 41/75\n",
            "64/64 [==============================] - 4s 70ms/step - loss: 0.0037 - accuracy: 0.9995\n",
            "Epoch 42/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 43/75\n",
            "64/64 [==============================] - 8s 124ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 44/75\n",
            "64/64 [==============================] - 5s 83ms/step - loss: 0.0025 - accuracy: 0.9995\n",
            "Epoch 45/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 46/75\n",
            "64/64 [==============================] - 5s 76ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 47/75\n",
            "64/64 [==============================] - 5s 80ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 48/75\n",
            "64/64 [==============================] - 7s 117ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 49/75\n",
            "64/64 [==============================] - 7s 101ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 50/75\n",
            "64/64 [==============================] - 4s 69ms/step - loss: 0.0026 - accuracy: 0.9995\n",
            "Epoch 51/75\n",
            "64/64 [==============================] - 5s 80ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 52/75\n",
            "64/64 [==============================] - 4s 69ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 53/75\n",
            "64/64 [==============================] - 5s 81ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 54/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 55/75\n",
            "64/64 [==============================] - 4s 70ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 56/75\n",
            "64/64 [==============================] - 5s 80ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 57/75\n",
            "64/64 [==============================] - 5s 75ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 58/75\n",
            "64/64 [==============================] - 5s 83ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 59/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 60/75\n",
            "64/64 [==============================] - 5s 71ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 61/75\n",
            "64/64 [==============================] - 5s 78ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 62/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 63/75\n",
            "64/64 [==============================] - 5s 78ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 64/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 65/75\n",
            "64/64 [==============================] - 5s 82ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 66/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 67/75\n",
            "64/64 [==============================] - 4s 69ms/step - loss: 0.0024 - accuracy: 0.9995\n",
            "Epoch 68/75\n",
            "64/64 [==============================] - 5s 76ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 69/75\n",
            "64/64 [==============================] - 5s 80ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 70/75\n",
            "64/64 [==============================] - 5s 83ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 71/75\n",
            "64/64 [==============================] - 4s 70ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 72/75\n",
            "64/64 [==============================] - 4s 70ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 73/75\n",
            "64/64 [==============================] - 5s 81ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 74/75\n",
            "64/64 [==============================] - 4s 68ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 75/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.0021 - accuracy: 0.9995\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7ed223584790>"
            ]
          },
          "metadata": {},
          "execution_count": 76
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W1nTP5DInSaT",
        "outputId": "f6174ee5-9c7b-40fa-9a5c-054ed55861fa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "32/32 [==============================] - 1s 12ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nFscUuqDnU8W",
        "outputId": "88ca9dda-5a1b-423f-9cda-849bb286eda3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9783889980353635,\n",
              " 0.9979919678714859,\n",
              " 0.9594594594594594,\n",
              " 0.9567945057489003,\n",
              " 0.9575334031178668)"
            ]
          },
          "metadata": {},
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oajMP36znYsY",
        "outputId": "94f7e4cc-0e85-462b-c58a-40160e30bcde"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9596153846153846"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "7cnBZ1S03laS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "iZZss04o3nzY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "ehEg9kN23uCy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "6ZMwLUyt3u7I"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "I2nS7l3q32xQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "4Nzsiz8B355g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "mOgHgLOd36zY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "Sey57gZs39EI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "5IQOps453_L4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "glQvEb5u4Crg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "1FYDyFfF4D15"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "KaxaGy574GMf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "ejswLMbF4IPP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DT1Q1W3m4Ky_",
        "outputId": "bd807908-0b1a-4e6e-c1b0-db5db1729710"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 7s 79ms/step - loss: 0.6371 - accuracy: 0.6411\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.5335 - accuracy: 0.7418\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.4620 - accuracy: 0.7846\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.4148 - accuracy: 0.8156\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.3911 - accuracy: 0.8197\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.3426 - accuracy: 0.8492\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.2987 - accuracy: 0.8756\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.2459 - accuracy: 0.8967\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.2046 - accuracy: 0.9117\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.1859 - accuracy: 0.9291\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 5s 83ms/step - loss: 0.1514 - accuracy: 0.9373\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.1096 - accuracy: 0.9615\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0851 - accuracy: 0.9700\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.1071 - accuracy: 0.9589\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 4s 69ms/step - loss: 0.0651 - accuracy: 0.9773\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0555 - accuracy: 0.9821\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0490 - accuracy: 0.9835\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0437 - accuracy: 0.9860\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0242 - accuracy: 0.9930\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0461 - accuracy: 0.9845\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0178 - accuracy: 0.9954\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0145 - accuracy: 0.9954\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0082 - accuracy: 0.9976\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.0131 - accuracy: 0.9964\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0121 - accuracy: 0.9964\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0043 - accuracy: 0.9990\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0039 - accuracy: 0.9988\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 5s 69ms/step - loss: 0.0046 - accuracy: 0.9985\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0077 - accuracy: 0.9978\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0061 - accuracy: 0.9983\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0549 - accuracy: 0.9811\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0275 - accuracy: 0.9901\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0478 - accuracy: 0.9864\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0256 - accuracy: 0.9923\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0133 - accuracy: 0.9954\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0189 - accuracy: 0.9937\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0067 - accuracy: 0.9978\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0053 - accuracy: 0.9981\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0048 - accuracy: 0.9985\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0085 - accuracy: 0.9973\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0036 - accuracy: 0.9993\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0056 - accuracy: 0.9983\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 5s 69ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0031 - accuracy: 0.9988\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0032 - accuracy: 0.9990\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 4s 67ms/step - loss: 0.0040 - accuracy: 0.9988\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0028 - accuracy: 0.9993\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0027 - accuracy: 0.9993\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0030 - accuracy: 0.9990\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.0028 - accuracy: 0.9993\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 5s 69ms/step - loss: 0.0024 - accuracy: 0.9993\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0026 - accuracy: 0.9995\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 5s 84ms/step - loss: 0.0024 - accuracy: 0.9993\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 5s 69ms/step - loss: 0.0032 - accuracy: 0.9993\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0026 - accuracy: 0.9993\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0026 - accuracy: 0.9993\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0026 - accuracy: 0.9993\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 5s 69ms/step - loss: 0.0025 - accuracy: 0.9993\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0023 - accuracy: 0.9993\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0022 - accuracy: 0.9993\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0028 - accuracy: 0.9993\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0025 - accuracy: 0.9993\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 6s 85ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 5s 85ms/step - loss: 0.0071 - accuracy: 0.9985\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.1124 - accuracy: 0.9664\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0435 - accuracy: 0.9857\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 5s 83ms/step - loss: 0.0131 - accuracy: 0.9966\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0653 - accuracy: 0.9831\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0126 - accuracy: 0.9969\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0064 - accuracy: 0.9988\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7ed220c04ac0>"
            ]
          },
          "metadata": {},
          "execution_count": 105
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "clK87E_S4NCn",
        "outputId": "8a790c76-7e3c-47bc-a29a-65b83f154fe6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 1s 12ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8Nq4DQHv4T9a",
        "outputId": "65da1719-f661-48fc-d674-cee36e09fd01"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9806201550387597,\n",
              " 1.0,\n",
              " 0.9628252788104089,\n",
              " 0.9612339038059892,\n",
              " 0.9619569949741276)"
            ]
          },
          "metadata": {},
          "execution_count": 107
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cs_3UWot4YAD",
        "outputId": "934bcc3b-27e3-4812-85da-ce9e88236034"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9610894941634242"
            ]
          },
          "metadata": {},
          "execution_count": 108
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "eRE0Bc1be7KK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "qFv2QMKBe9hX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "Ci5D37wLfAoP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "D59akdHffDt2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "d8OJRYuKfZE2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "JEiKZEFFfOJW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "NbDDCnS1fgKe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "oVMaGExVfi44"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "WKB08jbPfl3f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "Idjqs60AfoxI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "_2mV76GDfsKo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "97P9_YegfvJQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "G_znXYvNfyKN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3C_rFqS3f1aG",
        "outputId": "69794927-83de-4731-ce71-e479d4de0b06"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 7s 103ms/step - loss: 0.6576 - accuracy: 0.5349\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.5941 - accuracy: 0.6967\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.5678 - accuracy: 0.6957\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.5498 - accuracy: 0.6919\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.5320 - accuracy: 0.7190\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 2s 136ms/step - loss: 0.5563 - accuracy: 0.6967\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 3s 193ms/step - loss: 0.5115 - accuracy: 0.7316\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 2s 105ms/step - loss: 0.4914 - accuracy: 0.7519\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 2s 118ms/step - loss: 0.4755 - accuracy: 0.7810\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 3s 172ms/step - loss: 0.4934 - accuracy: 0.7703\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 2s 128ms/step - loss: 0.4720 - accuracy: 0.7829\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 2s 108ms/step - loss: 0.4681 - accuracy: 0.7868\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 3s 177ms/step - loss: 0.4582 - accuracy: 0.7965\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 2s 117ms/step - loss: 0.4468 - accuracy: 0.8004\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.4405 - accuracy: 0.8091\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.4203 - accuracy: 0.8304\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.4169 - accuracy: 0.8266\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.3964 - accuracy: 0.8236\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.3796 - accuracy: 0.8333\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 2s 147ms/step - loss: 0.3790 - accuracy: 0.8362\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 3s 152ms/step - loss: 0.3522 - accuracy: 0.8382\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.3948 - accuracy: 0.8246\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.3599 - accuracy: 0.8401\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.3398 - accuracy: 0.8450\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.3201 - accuracy: 0.8547\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.3224 - accuracy: 0.8391\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.2925 - accuracy: 0.8653\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 2s 141ms/step - loss: 0.2643 - accuracy: 0.8798\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 2s 87ms/step - loss: 0.2590 - accuracy: 0.8837\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.2562 - accuracy: 0.8798\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.3313 - accuracy: 0.8498\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.2793 - accuracy: 0.8760\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.2281 - accuracy: 0.9031\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.1976 - accuracy: 0.9147\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.2464 - accuracy: 0.8905\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 2s 135ms/step - loss: 0.1880 - accuracy: 0.9176\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 2s 100ms/step - loss: 0.2243 - accuracy: 0.9002\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.2755 - accuracy: 0.8682\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.1969 - accuracy: 0.9176\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.1622 - accuracy: 0.9331\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.1187 - accuracy: 0.9457\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.1074 - accuracy: 0.9545\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.1348 - accuracy: 0.9438\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 2s 127ms/step - loss: 0.1249 - accuracy: 0.9496\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 2s 109ms/step - loss: 0.1508 - accuracy: 0.9312\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0767 - accuracy: 0.9758\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0891 - accuracy: 0.9758\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.2025 - accuracy: 0.9176\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.1313 - accuracy: 0.9399\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.1066 - accuracy: 0.9593\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0777 - accuracy: 0.9680\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 2s 114ms/step - loss: 0.0557 - accuracy: 0.9864\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 2s 119ms/step - loss: 0.0292 - accuracy: 0.9952\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0342 - accuracy: 0.9864\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 2s 114ms/step - loss: 0.0193 - accuracy: 0.9942\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0416 - accuracy: 0.9864\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 82ms/step - loss: 0.0625 - accuracy: 0.9758\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0540 - accuracy: 0.9767\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0376 - accuracy: 0.9874\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 2s 128ms/step - loss: 0.0557 - accuracy: 0.9806\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 2s 107ms/step - loss: 0.0385 - accuracy: 0.9855\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0292 - accuracy: 0.9922\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.0375 - accuracy: 0.9932\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0597 - accuracy: 0.9835\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.0428 - accuracy: 0.9826\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 0.0368 - accuracy: 0.9903\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0296 - accuracy: 0.9913\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 2s 120ms/step - loss: 0.0096 - accuracy: 0.9981\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 2s 112ms/step - loss: 0.0061 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0015 - accuracy: 1.0000\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 83ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 8.2944e-04 - accuracy: 1.0000\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 6.4399e-04 - accuracy: 1.0000\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 1s 85ms/step - loss: 5.4288e-04 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7dd78f7830d0>"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J6V8vJxMf4iB",
        "outputId": "23fe0a5e-69b5-47dd-ba2e-90f816958571"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 13ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-Je6YymLf7kX",
        "outputId": "f672e267-66a2-4a6a-865f-d263b77d0101"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9108527131782945,\n",
              " 1.0,\n",
              " 0.8424657534246576,\n",
              " 0.8227916144060204,\n",
              " 0.8360230564938658)"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FrYjy852f_dm",
        "outputId": "7ca5548f-d657-4c19-edae-d1b0ebbf849b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8296296296296296"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CNN+LSTM(Geary)**"
      ],
      "metadata": {
        "id": "qbL9jyMhe6o2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "yLjMYM2lqa7r"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "H9gDxNtNqdIs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "f_ibqYdhqeiG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "XZLP7KaVqhOz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "EtC07snjqjI1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "QDDCtSCuqlqb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "iovMHrs9qncE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "ILwLz3cpqprs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "neTCw69JqruI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "7e0E4ZVYqtyj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "ia7-Tl70qwUs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "YpWXU_HCqyrc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3MQUBORGqzdT",
        "outputId": "3a694b8f-60eb-45d5-b59c-adce3eb747c4"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 5s 66ms/step - loss: 0.5282 - accuracy: 0.8005\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.5038 - accuracy: 0.8005\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.5026 - accuracy: 0.8005\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.5021 - accuracy: 0.8005\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.5068 - accuracy: 0.8005\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.5000 - accuracy: 0.8005\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.4994 - accuracy: 0.8005\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.4972 - accuracy: 0.8005\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.4951 - accuracy: 0.8005\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4906 - accuracy: 0.8005\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.4959 - accuracy: 0.8005\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 4s 90ms/step - loss: 0.4940 - accuracy: 0.8005\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4822 - accuracy: 0.8005\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4687 - accuracy: 0.8005\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4638 - accuracy: 0.8005\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 3s 78ms/step - loss: 0.4641 - accuracy: 0.8005\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 0.4576 - accuracy: 0.8009\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.4477 - accuracy: 0.8029\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4397 - accuracy: 0.8114\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.4306 - accuracy: 0.8215\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 0.4306 - accuracy: 0.8211\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4143 - accuracy: 0.8273\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.4084 - accuracy: 0.8269\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.3981 - accuracy: 0.8354\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.3903 - accuracy: 0.8354\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3872 - accuracy: 0.8439\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3637 - accuracy: 0.8563\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3522 - accuracy: 0.8548\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 3s 76ms/step - loss: 0.3346 - accuracy: 0.8590\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.3300 - accuracy: 0.8699\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.3242 - accuracy: 0.8722\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.2999 - accuracy: 0.8885\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.2979 - accuracy: 0.8761\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.2770 - accuracy: 0.8904\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.2524 - accuracy: 0.8966\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.2634 - accuracy: 0.8912\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.2278 - accuracy: 0.9098\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.2252 - accuracy: 0.9051\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.2260 - accuracy: 0.9051\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.1773 - accuracy: 0.9303\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.1724 - accuracy: 0.9284\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.2012 - accuracy: 0.9198\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 3s 69ms/step - loss: 0.1494 - accuracy: 0.9419\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.1208 - accuracy: 0.9543\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.1062 - accuracy: 0.9589\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.1159 - accuracy: 0.9566\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.0890 - accuracy: 0.9655\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0902 - accuracy: 0.9702\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0926 - accuracy: 0.9663\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0845 - accuracy: 0.9729\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0425 - accuracy: 0.9884\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0370 - accuracy: 0.9876\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0610 - accuracy: 0.9814\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0622 - accuracy: 0.9783\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0304 - accuracy: 0.9911\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0290 - accuracy: 0.9876\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0398 - accuracy: 0.9841\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0313 - accuracy: 0.9899\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0370 - accuracy: 0.9849\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0395 - accuracy: 0.9849\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0191 - accuracy: 0.9957\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0117 - accuracy: 0.9957\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 3s 79ms/step - loss: 0.0101 - accuracy: 0.9965\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 3s 67ms/step - loss: 0.0154 - accuracy: 0.9957\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0096 - accuracy: 0.9977\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0083 - accuracy: 0.9977\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.0087 - accuracy: 0.9965\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 3s 76ms/step - loss: 0.0051 - accuracy: 0.9977\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0097 - accuracy: 0.9965\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0070 - accuracy: 0.9973\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0058 - accuracy: 0.9969\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0050 - accuracy: 0.9981\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0096 - accuracy: 0.9961\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0140 - accuracy: 0.9954\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 3s 65ms/step - loss: 0.0254 - accuracy: 0.9915\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7dd77bf1bbe0>"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "89-6kf1Oq4LE",
        "outputId": "7db85869-525a-46af-bace-6af2edc9b07e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 1s 9ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f24fqmNiq5jb",
        "outputId": "2c2bdb12-37a3-42af-9e00-1c437bdc7435"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.958139534883721,\n",
              " 0.9307692307692308,\n",
              " 0.8705035971223022,\n",
              " 0.8732209805991337,\n",
              " 0.8740017604700463)"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WXToL20Uq9W8",
        "outputId": "593fcc7e-f3d8-4c47-8615-e5b6ed95ddd7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9650485436893204"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "K5-8o1EBpIFY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "r36hK3_XpJ3c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "-cfmJaC3pxTC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "CIoR6oGjp2Mw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "RXXf5YuAp7et"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "MdxZsxGGp-Lj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "# cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "241w2s61qAZk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "kmFmgUV7qC9q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "W48whAkoqE9R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "OdUcM2eIqG6e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "qVtPeyLSqI-g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "faZxi3fjqLAt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L3nWrztHqNJ9",
        "outputId": "88cd2469-9b52-4257-8993-08ceff3b8f64"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 5s 74ms/step - loss: 0.5226 - accuracy: 0.7883\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.4925 - accuracy: 0.8087\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 3s 73ms/step - loss: 0.4913 - accuracy: 0.8087\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.4894 - accuracy: 0.8087\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.4916 - accuracy: 0.8087\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.4897 - accuracy: 0.8087\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 3s 76ms/step - loss: 0.4881 - accuracy: 0.8087\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 3s 73ms/step - loss: 0.4852 - accuracy: 0.8087\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.4879 - accuracy: 0.8087\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.4823 - accuracy: 0.8087\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.4758 - accuracy: 0.8087\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 2s 69ms/step - loss: 0.4711 - accuracy: 0.8087\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 0.4703 - accuracy: 0.8096\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 3s 76ms/step - loss: 0.4717 - accuracy: 0.8096\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 0.4667 - accuracy: 0.8096\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 2s 67ms/step - loss: 0.4639 - accuracy: 0.8100\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 3s 76ms/step - loss: 0.4689 - accuracy: 0.8096\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.4576 - accuracy: 0.8096\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.4559 - accuracy: 0.8096\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 2s 62ms/step - loss: 0.4578 - accuracy: 0.8096\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.4448 - accuracy: 0.8060\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 0.4504 - accuracy: 0.8096\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.4391 - accuracy: 0.8096\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.4306 - accuracy: 0.8109\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.4278 - accuracy: 0.8144\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.4169 - accuracy: 0.8153\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 2s 47ms/step - loss: 0.4150 - accuracy: 0.8224\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.4110 - accuracy: 0.8251\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.4080 - accuracy: 0.8264\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.4089 - accuracy: 0.8175\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.4001 - accuracy: 0.8291\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.3852 - accuracy: 0.8397\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.3764 - accuracy: 0.8450\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 2s 49ms/step - loss: 0.3608 - accuracy: 0.8530\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 2s 56ms/step - loss: 0.3541 - accuracy: 0.8578\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.3470 - accuracy: 0.8601\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.3306 - accuracy: 0.8667\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.3106 - accuracy: 0.8751\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.2966 - accuracy: 0.8782\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.2854 - accuracy: 0.8875\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.2733 - accuracy: 0.8906\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.2438 - accuracy: 0.9026\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.2301 - accuracy: 0.9092\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.2094 - accuracy: 0.9136\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.2145 - accuracy: 0.9167\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.1795 - accuracy: 0.9265\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.1659 - accuracy: 0.9358\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.1795 - accuracy: 0.9318\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 2s 51ms/step - loss: 0.1332 - accuracy: 0.9477\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.1366 - accuracy: 0.9473\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.1499 - accuracy: 0.9358\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0971 - accuracy: 0.9615\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.1037 - accuracy: 0.9637\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0835 - accuracy: 0.9668\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 2s 56ms/step - loss: 0.0961 - accuracy: 0.9641\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 2s 50ms/step - loss: 0.0812 - accuracy: 0.9672\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.0664 - accuracy: 0.9734\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0636 - accuracy: 0.9779\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0600 - accuracy: 0.9783\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.0444 - accuracy: 0.9845\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0296 - accuracy: 0.9911\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 2s 58ms/step - loss: 0.0269 - accuracy: 0.9907\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 2s 48ms/step - loss: 0.0225 - accuracy: 0.9925\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 2s 44ms/step - loss: 0.0499 - accuracy: 0.9823\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.0299 - accuracy: 0.9907\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0179 - accuracy: 0.9942\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0145 - accuracy: 0.9942\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0138 - accuracy: 0.9956\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0081 - accuracy: 0.9978\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.0148 - accuracy: 0.9960\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0079 - accuracy: 0.9969\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 2s 46ms/step - loss: 0.0112 - accuracy: 0.9960\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0132 - accuracy: 0.9956\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0163 - accuracy: 0.9938\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 2s 45ms/step - loss: 0.0337 - accuracy: 0.9885\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x795a8c938580>"
            ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O4sf_3Z2qQJQ",
        "outputId": "5010af24-4cea-4e22-fa72-b85eb1cf4417"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lzWbbqOpqTnx",
        "outputId": "ec2d4b70-6bba-49c2-c2a4-6412fcc1caf5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8802889576883385,\n",
              " 0.6678200692041523,\n",
              " 0.9061032863849765,\n",
              " 0.7689243027888446,\n",
              " 0.690623245367771,\n",
              " 0.7052631718409899)"
            ]
          },
          "metadata": {},
          "execution_count": 77
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yWTlAz_GqWB7",
        "outputId": "c34cd7f9-df84-4fc7-ec6a-40e939e8d3ae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.873015873015873"
            ]
          },
          "metadata": {},
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "KkX8OlW3kmMQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "LhRVQ2iQe8vu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "4UMPv6_rfB09"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "qI9i49GJg64l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "_vxdVRskfEum"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "jealvYeMfINf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "_DW2nf7FfLMX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "sRJpzFN5fNvG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "JasgppixfQpf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "_FmZ8IOpfTg4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "faZaLvLRfZaO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "v35zEYOCfZ4V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "sqVO_B9hfchG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 32)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EbKZvQxZfe4V",
        "outputId": "27e1afd6-2af1-42b8-e60f-fe285c583e0a"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/75\n",
            "130/130 [==============================] - 8s 40ms/step - loss: 0.6940 - accuracy: 0.4881\n",
            "Epoch 2/75\n",
            "130/130 [==============================] - 6s 48ms/step - loss: 0.6932 - accuracy: 0.4857\n",
            "Epoch 3/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.6928 - accuracy: 0.5083\n",
            "Epoch 4/75\n",
            "130/130 [==============================] - 6s 48ms/step - loss: 0.6920 - accuracy: 0.5228\n",
            "Epoch 5/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.6879 - accuracy: 0.5336\n",
            "Epoch 6/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.6838 - accuracy: 0.5474\n",
            "Epoch 7/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.6829 - accuracy: 0.5563\n",
            "Epoch 8/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.6728 - accuracy: 0.5775\n",
            "Epoch 9/75\n",
            "130/130 [==============================] - 8s 58ms/step - loss: 0.6636 - accuracy: 0.5883\n",
            "Epoch 10/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.6474 - accuracy: 0.6127\n",
            "Epoch 11/75\n",
            "130/130 [==============================] - 6s 48ms/step - loss: 0.6331 - accuracy: 0.6279\n",
            "Epoch 12/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.6162 - accuracy: 0.6462\n",
            "Epoch 13/75\n",
            "130/130 [==============================] - 6s 48ms/step - loss: 0.5753 - accuracy: 0.6884\n",
            "Epoch 14/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.5454 - accuracy: 0.7117\n",
            "Epoch 15/75\n",
            "130/130 [==============================] - 6s 48ms/step - loss: 0.4961 - accuracy: 0.7551\n",
            "Epoch 16/75\n",
            "130/130 [==============================] - 5s 41ms/step - loss: 0.4137 - accuracy: 0.8012\n",
            "Epoch 17/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.3625 - accuracy: 0.8327\n",
            "Epoch 18/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.3410 - accuracy: 0.8498\n",
            "Epoch 19/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.2721 - accuracy: 0.8870\n",
            "Epoch 20/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.2432 - accuracy: 0.9009\n",
            "Epoch 21/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.1750 - accuracy: 0.9335\n",
            "Epoch 22/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.1647 - accuracy: 0.9402\n",
            "Epoch 23/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.1395 - accuracy: 0.9438\n",
            "Epoch 24/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.1136 - accuracy: 0.9610\n",
            "Epoch 25/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.0839 - accuracy: 0.9718\n",
            "Epoch 26/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.0679 - accuracy: 0.9776\n",
            "Epoch 27/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0799 - accuracy: 0.9720\n",
            "Epoch 28/75\n",
            "130/130 [==============================] - 6s 44ms/step - loss: 0.0639 - accuracy: 0.9771\n",
            "Epoch 29/75\n",
            "130/130 [==============================] - 5s 41ms/step - loss: 0.0371 - accuracy: 0.9884\n",
            "Epoch 30/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0245 - accuracy: 0.9925\n",
            "Epoch 31/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.0263 - accuracy: 0.9918\n",
            "Epoch 32/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.0548 - accuracy: 0.9824\n",
            "Epoch 33/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.0369 - accuracy: 0.9884\n",
            "Epoch 34/75\n",
            "130/130 [==============================] - 5s 39ms/step - loss: 0.0526 - accuracy: 0.9819\n",
            "Epoch 35/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0289 - accuracy: 0.9918\n",
            "Epoch 36/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0137 - accuracy: 0.9964\n",
            "Epoch 37/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0069 - accuracy: 0.9988\n",
            "Epoch 38/75\n",
            "130/130 [==============================] - 5s 41ms/step - loss: 0.0062 - accuracy: 0.9988\n",
            "Epoch 39/75\n",
            "130/130 [==============================] - 6s 47ms/step - loss: 0.0047 - accuracy: 0.9988\n",
            "Epoch 40/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0043 - accuracy: 0.9988\n",
            "Epoch 41/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.0027 - accuracy: 0.9993\n",
            "Epoch 42/75\n",
            "130/130 [==============================] - 5s 41ms/step - loss: 0.0036 - accuracy: 0.9993\n",
            "Epoch 43/75\n",
            "130/130 [==============================] - 5s 42ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 44/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.0050 - accuracy: 0.9986\n",
            "Epoch 45/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0035 - accuracy: 0.9988\n",
            "Epoch 46/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0080 - accuracy: 0.9976\n",
            "Epoch 47/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.1200 - accuracy: 0.9670\n",
            "Epoch 48/75\n",
            "130/130 [==============================] - 6s 45ms/step - loss: 0.0322 - accuracy: 0.9908\n",
            "Epoch 49/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0256 - accuracy: 0.9925\n",
            "Epoch 50/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0227 - accuracy: 0.9933\n",
            "Epoch 51/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0235 - accuracy: 0.9928\n",
            "Epoch 52/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0084 - accuracy: 0.9983\n",
            "Epoch 53/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0039 - accuracy: 0.9993\n",
            "Epoch 54/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0038 - accuracy: 0.9993\n",
            "Epoch 55/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0027 - accuracy: 0.9988\n",
            "Epoch 56/75\n",
            "130/130 [==============================] - 6s 43ms/step - loss: 0.0027 - accuracy: 0.9990\n",
            "Epoch 57/75\n",
            "130/130 [==============================] - 6s 43ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 58/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0029 - accuracy: 0.9990\n",
            "Epoch 59/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0021 - accuracy: 0.9995\n",
            "Epoch 60/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0037 - accuracy: 0.9986\n",
            "Epoch 61/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0038 - accuracy: 0.9986\n",
            "Epoch 62/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0034 - accuracy: 0.9993\n",
            "Epoch 63/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0029 - accuracy: 0.9988\n",
            "Epoch 64/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0987 - accuracy: 0.9706\n",
            "Epoch 65/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0626 - accuracy: 0.9812\n",
            "Epoch 66/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0173 - accuracy: 0.9954\n",
            "Epoch 67/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0146 - accuracy: 0.9957\n",
            "Epoch 68/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0267 - accuracy: 0.9901\n",
            "Epoch 69/75\n",
            "130/130 [==============================] - 6s 44ms/step - loss: 0.0079 - accuracy: 0.9971\n",
            "Epoch 70/75\n",
            "130/130 [==============================] - 6s 42ms/step - loss: 0.0039 - accuracy: 0.9993\n",
            "Epoch 71/75\n",
            "130/130 [==============================] - 5s 41ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 72/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0032 - accuracy: 0.9990\n",
            "Epoch 73/75\n",
            "130/130 [==============================] - 5s 41ms/step - loss: 0.0030 - accuracy: 0.9990\n",
            "Epoch 74/75\n",
            "130/130 [==============================] - 6s 46ms/step - loss: 0.0030 - accuracy: 0.9990\n",
            "Epoch 75/75\n",
            "130/130 [==============================] - 5s 40ms/step - loss: 0.0027 - accuracy: 0.9993\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7dd77bb47490>"
            ]
          },
          "execution_count": 50,
          "metadata": {},
          "output_type": "execute_result"
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N1Fg1mfKfh-N",
        "outputId": "16fb1277-2695-4b63-8857-b0089883866e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 1s 15ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kPiuva74fkAW",
        "outputId": "8f2324aa-6fcb-4e9c-a22d-504fa77e342f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9710703953712633,\n",
              " 0.9981549815498155,\n",
              " 0.9491228070175438,\n",
              " 0.9418791474861004,\n",
              " 0.9432680202666277)"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D-x_ZlP_fm6Q",
        "outputId": "ad44d073-cf96-4170-9d59-e730185fa7a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9414141414141414"
            ]
          },
          "metadata": {},
          "execution_count": 53
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "bgUoOxUwoUK1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "752qFwrqoWHS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "lNKTTV-AoZKt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "UBATppjjocLd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "0k2Z9isWoj69"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "zJmQwl6ponUt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "8oAN7EPVoqRW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "_93hVTLcos_F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "-awhE_bLowF4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "fsq0OU-vozAl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "ZfXmRvqYo1_X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "KiJhe11Eo41N"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "ZpzhNaW_o7t9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gih-kDXHo_iV",
        "outputId": "27119f17-b08d-4c78-e26c-4c72ca0cfec6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 9s 93ms/step - loss: 0.6939 - accuracy: 0.5068\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 6s 85ms/step - loss: 0.6929 - accuracy: 0.5075\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 7s 106ms/step - loss: 0.6925 - accuracy: 0.5162\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 5s 80ms/step - loss: 0.6896 - accuracy: 0.5426\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 6s 92ms/step - loss: 0.6898 - accuracy: 0.5358\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.6741 - accuracy: 0.5828\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 4s 67ms/step - loss: 0.6428 - accuracy: 0.6254\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.6467 - accuracy: 0.6171\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.6065 - accuracy: 0.6607\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.5954 - accuracy: 0.6755\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.5827 - accuracy: 0.6868\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.5422 - accuracy: 0.7268\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.5244 - accuracy: 0.7413\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 5s 83ms/step - loss: 0.4557 - accuracy: 0.7773\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.4288 - accuracy: 0.7982\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.3956 - accuracy: 0.8122\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.3560 - accuracy: 0.8371\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.3132 - accuracy: 0.8674\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.3250 - accuracy: 0.8606\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.2855 - accuracy: 0.8797\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.2264 - accuracy: 0.9092\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.2103 - accuracy: 0.9153\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.1810 - accuracy: 0.9279\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 4s 69ms/step - loss: 0.1891 - accuracy: 0.9274\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.1500 - accuracy: 0.9402\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.1528 - accuracy: 0.9429\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.1199 - accuracy: 0.9511\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 5s 69ms/step - loss: 0.0912 - accuracy: 0.9666\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.1216 - accuracy: 0.9567\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0932 - accuracy: 0.9644\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0849 - accuracy: 0.9652\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0705 - accuracy: 0.9760\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0722 - accuracy: 0.9739\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0525 - accuracy: 0.9828\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0438 - accuracy: 0.9860\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0482 - accuracy: 0.9833\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0559 - accuracy: 0.9818\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0498 - accuracy: 0.9816\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0294 - accuracy: 0.9908\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0177 - accuracy: 0.9942\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0320 - accuracy: 0.9889\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0259 - accuracy: 0.9925\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0577 - accuracy: 0.9785\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0706 - accuracy: 0.9773\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0164 - accuracy: 0.9964\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0117 - accuracy: 0.9961\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0237 - accuracy: 0.9918\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0168 - accuracy: 0.9959\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0228 - accuracy: 0.9925\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0301 - accuracy: 0.9901\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0578 - accuracy: 0.9818\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0511 - accuracy: 0.9809\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0167 - accuracy: 0.9959\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0051 - accuracy: 0.9990\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0045 - accuracy: 0.9990\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0047 - accuracy: 0.9988\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 5s 83ms/step - loss: 0.0394 - accuracy: 0.9889\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0158 - accuracy: 0.9956\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0081 - accuracy: 0.9978\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 5s 81ms/step - loss: 0.0390 - accuracy: 0.9848\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0156 - accuracy: 0.9949\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0861 - accuracy: 0.9739\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0174 - accuracy: 0.9954\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0153 - accuracy: 0.9971\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0069 - accuracy: 0.9983\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0045 - accuracy: 0.9988\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0043 - accuracy: 0.9988\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 5s 83ms/step - loss: 0.0041 - accuracy: 0.9988\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0040 - accuracy: 0.9990\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0031 - accuracy: 0.9993\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 5s 82ms/step - loss: 0.0026 - accuracy: 0.9993\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0032 - accuracy: 0.9993\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0028 - accuracy: 0.9993\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0073 - accuracy: 0.9981\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7dd7797f6260>"
            ]
          },
          "metadata": {},
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LH7se4wipEUO",
        "outputId": "4484733a-487f-4521-d181-2e8b6e710867"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 8ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rRbIZ7RfpGrJ",
        "outputId": "1017f0ee-90a9-46c4-ccc4-f2a070ddae12"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9573643410852714,\n",
              " 0.9880952380952381,\n",
              " 0.9291044776119403,\n",
              " 0.9148054755043228,\n",
              " 0.9165665161407893)"
            ]
          },
          "metadata": {},
          "execution_count": 68
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gpWAKoTmpKVV",
        "outputId": "7c05c8a6-2bd6-4ae9-dc53-fe69579a2492"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.928030303030303"
            ]
          },
          "metadata": {},
          "execution_count": 69
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "vE29SQlCnDbF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "BMiYKYIfnGg_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "m-yfW4-pnRXg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "WzTGxjNInZ7d"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "wt6RyD-bnhjd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "Oda9HQnanl0G"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "aj-4GTaJnpM0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "#cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "ImwpYhyxnsu2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "1zSBzpvnnv5N"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(LSTM(128, activation='relu'))"
      ],
      "metadata": {
        "id": "PQrLSucznzNY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "-GP_3RVpn2JO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "AjaNi0j_n5BF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "x_1X8IJAn8Fd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Qym4y8cSn_Ac",
        "outputId": "19a271b7-60a3-4471-d932-df411ac4c5b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 3s 61ms/step - loss: 0.6899 - accuracy: 0.5281\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.6460 - accuracy: 0.5940\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.5615 - accuracy: 0.7229\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4868 - accuracy: 0.7742\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 77ms/step - loss: 0.5100 - accuracy: 0.7558\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 2s 103ms/step - loss: 0.4732 - accuracy: 0.7762\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 2s 107ms/step - loss: 0.4756 - accuracy: 0.7771\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 2s 101ms/step - loss: 0.4714 - accuracy: 0.7810\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4721 - accuracy: 0.7800\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4625 - accuracy: 0.7849\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4564 - accuracy: 0.7781\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4445 - accuracy: 0.7926\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4680 - accuracy: 0.7955\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4583 - accuracy: 0.7878\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4428 - accuracy: 0.8081\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4585 - accuracy: 0.7936\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 1s 62ms/step - loss: 0.4336 - accuracy: 0.8052\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 2s 93ms/step - loss: 0.4545 - accuracy: 0.7878\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 86ms/step - loss: 0.4440 - accuracy: 0.7984\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.4769 - accuracy: 0.7713\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.4477 - accuracy: 0.7946\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4230 - accuracy: 0.8081\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4204 - accuracy: 0.8101\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.4261 - accuracy: 0.8072\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4331 - accuracy: 0.7994\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4621 - accuracy: 0.7917\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4861 - accuracy: 0.7645\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4339 - accuracy: 0.8120\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 1s 80ms/step - loss: 0.4314 - accuracy: 0.7975\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 2s 99ms/step - loss: 0.4304 - accuracy: 0.8014\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 63ms/step - loss: 0.4201 - accuracy: 0.8081\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4201 - accuracy: 0.8062\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4211 - accuracy: 0.8004\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4153 - accuracy: 0.8140\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4236 - accuracy: 0.8110\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4311 - accuracy: 0.8072\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 1s 62ms/step - loss: 0.4213 - accuracy: 0.8023\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4074 - accuracy: 0.8149\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4243 - accuracy: 0.8043\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 1s 64ms/step - loss: 0.4170 - accuracy: 0.8014\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 2s 97ms/step - loss: 0.4410 - accuracy: 0.7926\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 79ms/step - loss: 0.4334 - accuracy: 0.7994\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.4082 - accuracy: 0.8217\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4058 - accuracy: 0.8159\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4035 - accuracy: 0.8130\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4026 - accuracy: 0.8149\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4048 - accuracy: 0.8178\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3921 - accuracy: 0.8169\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.4296 - accuracy: 0.8033\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3967 - accuracy: 0.8169\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3953 - accuracy: 0.8130\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 2s 89ms/step - loss: 0.3920 - accuracy: 0.8198\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 2s 95ms/step - loss: 0.3960 - accuracy: 0.8110\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4026 - accuracy: 0.8140\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4015 - accuracy: 0.8043\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3928 - accuracy: 0.8227\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4166 - accuracy: 0.7897\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3883 - accuracy: 0.8236\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4037 - accuracy: 0.8178\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3825 - accuracy: 0.8275\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3853 - accuracy: 0.8295\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3952 - accuracy: 0.8159\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.3810 - accuracy: 0.8266\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 2s 95ms/step - loss: 0.3647 - accuracy: 0.8285\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 75ms/step - loss: 0.3594 - accuracy: 0.8314\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.3920 - accuracy: 0.8062\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3526 - accuracy: 0.8343\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.3758 - accuracy: 0.8149\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.3765 - accuracy: 0.8295\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.3681 - accuracy: 0.8333\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3488 - accuracy: 0.8421\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.3422 - accuracy: 0.8401\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 1s 61ms/step - loss: 0.3330 - accuracy: 0.8401\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.3347 - accuracy: 0.8576\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 2s 95ms/step - loss: 0.3280 - accuracy: 0.8605\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7ea3e45d3490>"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7Ld9P3bBoCt3",
        "outputId": "c7d5e777-477b-480b-a3e7-91525e025295"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vn-jsQ-zoJos",
        "outputId": "2b2fcbb2-2c84-47ea-f924-477481404058"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8488372093023255,\n",
              " 0.832,\n",
              " 0.8524590163934426,\n",
              " 0.6971648708842473,\n",
              " 0.6973541087749341)"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_uJZtjCFoKRw",
        "outputId": "7f42c5a8-8aa3-4706-88ce-d3130aed93c5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8646616541353384"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CNN(APAAC)**"
      ],
      "metadata": {
        "id": "zn0iRJUc6DVR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "AWjOUp_Xhsxb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "SyD9aswX6Hk5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "2MSk20v06PJ6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "RNKrUhy16UrZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "pnjlh3Vp6Wda"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "jViEGWaq6Ym5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "dabdXTDy6ceT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "MkRqMe8Q6eY6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "lNXiUHvG6ggp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "WSjNRHeo6ika"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "49k_G49_6kMU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4hIpHxf36mSc",
        "outputId": "47be9a6d-df7b-4d7e-b13e-abd57f5eb7fa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 9s 119ms/step - loss: 0.4335 - accuracy: 0.7971\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 7s 174ms/step - loss: 0.3550 - accuracy: 0.8517\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 5s 113ms/step - loss: 0.3286 - accuracy: 0.8629\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 5s 115ms/step - loss: 0.2934 - accuracy: 0.8799\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 6s 152ms/step - loss: 0.2815 - accuracy: 0.8857\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 4s 99ms/step - loss: 0.2696 - accuracy: 0.8877\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 6s 138ms/step - loss: 0.2374 - accuracy: 0.9040\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.2134 - accuracy: 0.9117\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 3s 77ms/step - loss: 0.1933 - accuracy: 0.9175\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 5s 134ms/step - loss: 0.1625 - accuracy: 0.9345\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 5s 125ms/step - loss: 0.1379 - accuracy: 0.9473\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 4s 93ms/step - loss: 0.1330 - accuracy: 0.9450\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 6s 140ms/step - loss: 0.1007 - accuracy: 0.9640\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 3s 80ms/step - loss: 0.0828 - accuracy: 0.9717\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 4s 104ms/step - loss: 0.0843 - accuracy: 0.9710\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 7s 166ms/step - loss: 0.0620 - accuracy: 0.9814\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 5s 128ms/step - loss: 0.0842 - accuracy: 0.9690\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 6s 138ms/step - loss: 0.0810 - accuracy: 0.9694\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 4s 109ms/step - loss: 0.0572 - accuracy: 0.9822\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0411 - accuracy: 0.9899\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 5s 135ms/step - loss: 0.0314 - accuracy: 0.9923\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.0246 - accuracy: 0.9923\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0294 - accuracy: 0.9911\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 6s 151ms/step - loss: 0.0321 - accuracy: 0.9903\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 4s 108ms/step - loss: 0.0336 - accuracy: 0.9911\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 5s 127ms/step - loss: 0.0170 - accuracy: 0.9961\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 5s 115ms/step - loss: 0.0137 - accuracy: 0.9965\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.0127 - accuracy: 0.9969\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 5s 119ms/step - loss: 0.0121 - accuracy: 0.9965\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0245 - accuracy: 0.9942\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 5s 111ms/step - loss: 0.0303 - accuracy: 0.9899\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 4s 96ms/step - loss: 0.0226 - accuracy: 0.9926\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.0227 - accuracy: 0.9923\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 0.0115 - accuracy: 0.9969\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 5s 116ms/step - loss: 0.0167 - accuracy: 0.9942\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 5s 120ms/step - loss: 0.0127 - accuracy: 0.9950\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 3s 73ms/step - loss: 0.0119 - accuracy: 0.9961\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 0.0138 - accuracy: 0.9965\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 6s 158ms/step - loss: 0.0117 - accuracy: 0.9965\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 5s 120ms/step - loss: 0.0159 - accuracy: 0.9961\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 7s 172ms/step - loss: 0.0213 - accuracy: 0.9942\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 5s 130ms/step - loss: 0.0153 - accuracy: 0.9942\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 7s 177ms/step - loss: 0.0084 - accuracy: 0.9985\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 6s 137ms/step - loss: 0.0082 - accuracy: 0.9977\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 6s 146ms/step - loss: 0.0057 - accuracy: 0.9977\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 5s 118ms/step - loss: 0.0078 - accuracy: 0.9973\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 4s 102ms/step - loss: 0.0180 - accuracy: 0.9938\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 6s 156ms/step - loss: 0.0299 - accuracy: 0.9907\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.0283 - accuracy: 0.9919\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 6s 142ms/step - loss: 0.0404 - accuracy: 0.9876\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.0463 - accuracy: 0.9833\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 4s 106ms/step - loss: 0.0304 - accuracy: 0.9899\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 5s 131ms/step - loss: 0.0145 - accuracy: 0.9954\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 4s 105ms/step - loss: 0.0186 - accuracy: 0.9961\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0107 - accuracy: 0.9981\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 6s 138ms/step - loss: 0.0116 - accuracy: 0.9969\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 4s 109ms/step - loss: 0.0090 - accuracy: 0.9977\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 4s 106ms/step - loss: 0.0116 - accuracy: 0.9977\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 5s 117ms/step - loss: 0.0065 - accuracy: 0.9977\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 4s 99ms/step - loss: 0.0060 - accuracy: 0.9973\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 5s 130ms/step - loss: 0.0063 - accuracy: 0.9981\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 5s 129ms/step - loss: 0.0068 - accuracy: 0.9977\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 6s 142ms/step - loss: 0.0105 - accuracy: 0.9969\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 6s 151ms/step - loss: 0.0114 - accuracy: 0.9969\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 6s 141ms/step - loss: 0.0105 - accuracy: 0.9973\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 6s 142ms/step - loss: 0.0113 - accuracy: 0.9969\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 5s 112ms/step - loss: 0.0115 - accuracy: 0.9981\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 5s 116ms/step - loss: 0.0127 - accuracy: 0.9985\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 7s 163ms/step - loss: 0.0051 - accuracy: 0.9981\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 4s 109ms/step - loss: 0.0057 - accuracy: 0.9981\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 6s 152ms/step - loss: 0.0076 - accuracy: 0.9973\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 5s 113ms/step - loss: 0.0049 - accuracy: 0.9977\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 5s 131ms/step - loss: 0.0040 - accuracy: 0.9988\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 6s 138ms/step - loss: 0.0031 - accuracy: 0.9992\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 5s 113ms/step - loss: 0.0088 - accuracy: 0.9985\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23cf2770a0>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LWjlb6sU6ouA",
        "outputId": "dca0ea3f-4e6d-4d9f-bfb1-2499aa1fc409"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 0s 14ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cY3QDuYu6rF5",
        "outputId": "601a92d7-3f5b-4dd3-8666-190b0f14fad5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9658914728682171,\n",
              " 0.9318181818181818,\n",
              " 0.9044117647058824,\n",
              " 0.8963900814860247,\n",
              " 0.8965491777354397)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VNjUOcrK6tc4",
        "outputId": "20b01ec6-d127-403d-ab2a-469f2e83ffea"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9746588693957114"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "Izsoxq_jhyZF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "waYrA7oUhz-n"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "7J_VSA23h5Yr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "yR9llpu-h7it"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "LoJ6EpHBh95J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "HlKeuvzhh__Y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "D9Sjf_uoiH0o"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "U8Jd9hapiezB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "MJj8feDAihpA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "jgkhoVd3ikc_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "F9xLV4lKinyv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V8ySu3K_irEP",
        "outputId": "2e1d05a1-5520-4bc6-d571-06199c5cd0dc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 4s 76ms/step - loss: 0.4179 - accuracy: 0.8074\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 3s 92ms/step - loss: 0.3454 - accuracy: 0.8605\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 3s 70ms/step - loss: 0.3147 - accuracy: 0.8738\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 3s 73ms/step - loss: 0.3096 - accuracy: 0.8804\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.2735 - accuracy: 0.8964\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 3s 81ms/step - loss: 0.2492 - accuracy: 0.8999\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 3s 78ms/step - loss: 0.2309 - accuracy: 0.9043\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 2s 69ms/step - loss: 0.2242 - accuracy: 0.9061\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 2s 67ms/step - loss: 0.1942 - accuracy: 0.9167\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 3s 89ms/step - loss: 0.1789 - accuracy: 0.9296\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 4s 101ms/step - loss: 0.1529 - accuracy: 0.9389\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.1385 - accuracy: 0.9460\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.1280 - accuracy: 0.9491\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.1068 - accuracy: 0.9544\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 3s 94ms/step - loss: 0.0907 - accuracy: 0.9708\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0721 - accuracy: 0.9730\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0736 - accuracy: 0.9717\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 3s 73ms/step - loss: 0.0551 - accuracy: 0.9823\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 4s 99ms/step - loss: 0.0395 - accuracy: 0.9903\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 3s 77ms/step - loss: 0.0266 - accuracy: 0.9925\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0242 - accuracy: 0.9947\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 3s 87ms/step - loss: 0.0415 - accuracy: 0.9863\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 3s 90ms/step - loss: 0.0451 - accuracy: 0.9858\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.0393 - accuracy: 0.9876\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 3s 75ms/step - loss: 0.0278 - accuracy: 0.9929\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.0383 - accuracy: 0.9880\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 3s 78ms/step - loss: 0.0500 - accuracy: 0.9925\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 3s 83ms/step - loss: 0.0171 - accuracy: 0.9969\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0110 - accuracy: 0.9978\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0088 - accuracy: 0.9978\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0126 - accuracy: 0.9965\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0096 - accuracy: 0.9978\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 2s 69ms/step - loss: 0.0076 - accuracy: 0.9973\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0077 - accuracy: 0.9987\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0078 - accuracy: 0.9982\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0131 - accuracy: 0.9969\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0321 - accuracy: 0.9942\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0223 - accuracy: 0.9938\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.0089 - accuracy: 0.9982\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0093 - accuracy: 0.9978\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.0044 - accuracy: 0.9987\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0216 - accuracy: 0.9951\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0115 - accuracy: 0.9973\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0088 - accuracy: 0.9982\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 2s 70ms/step - loss: 0.0036 - accuracy: 0.9991\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.0068 - accuracy: 0.9973\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0143 - accuracy: 0.9969\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0063 - accuracy: 0.9978\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.0090 - accuracy: 0.9978\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0392 - accuracy: 0.9880\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 3s 70ms/step - loss: 0.0849 - accuracy: 0.9699\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0333 - accuracy: 0.9889\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0129 - accuracy: 0.9965\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.0086 - accuracy: 0.9987\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0106 - accuracy: 0.9969\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.0057 - accuracy: 0.9982\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 2s 68ms/step - loss: 0.0059 - accuracy: 0.9973\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 2s 52ms/step - loss: 0.0047 - accuracy: 0.9991\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.0058 - accuracy: 0.9978\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0075 - accuracy: 0.9982\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.0032 - accuracy: 0.9987\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 2s 56ms/step - loss: 0.0035 - accuracy: 0.9987\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 2s 67ms/step - loss: 0.0021 - accuracy: 0.9987\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0046 - accuracy: 0.9982\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0030 - accuracy: 0.9991\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0079 - accuracy: 0.9978\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0026 - accuracy: 0.9987\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.0038 - accuracy: 0.9987\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 2s 63ms/step - loss: 0.0027 - accuracy: 0.9991\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0026 - accuracy: 0.9982\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0025 - accuracy: 0.9987\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0037 - accuracy: 0.9978\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0039 - accuracy: 0.9991\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.0046 - accuracy: 0.9982\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.0029 - accuracy: 0.9987\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x795a91554940>"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EF-TbeNiiuBA",
        "outputId": "f9514214-6814-4dca-f570-c6804a7a4c5d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GMqGBhV7i0jf",
        "outputId": "5f4db972-f005-48a1-ae21-256de20854ed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9453044375644994,\n",
              " 0.8506224066390041,\n",
              " 0.9234234234234234,\n",
              " 0.8855291576673866,\n",
              " 0.8496764166103213,\n",
              " 0.8509128777007038)"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6zBX2R90i5Rk",
        "outputId": "71a7c0ef-4ef7-435f-a60a-06692143cecc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9518072289156626"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "wocis_xY6vvM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "HsXdKBLo6x0g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "yfVx5Jrg616b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "5wOy-Kr964b5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "3Lj-udQE66OZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "2fVjlEUB67w6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "0bwLdoUq69oQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "G9V--hzh6_QI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "xA-BwbWW7BB5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "rTgi76Qy7CiA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "JepgmLV17EE7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "46ZmZtkx7FzB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dww2MkaI7Hlg",
        "outputId": "168711fc-f151-4863-e62d-34d7b1054221"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "64/64 [==============================] - 7s 81ms/step - loss: 0.5348 - accuracy: 0.7289\n",
            "Epoch 2/75\n",
            "64/64 [==============================] - 5s 79ms/step - loss: 0.4022 - accuracy: 0.8197\n",
            "Epoch 3/75\n",
            "64/64 [==============================] - 6s 94ms/step - loss: 0.2783 - accuracy: 0.8954\n",
            "Epoch 4/75\n",
            "64/64 [==============================] - 5s 76ms/step - loss: 0.2064 - accuracy: 0.9251\n",
            "Epoch 5/75\n",
            "64/64 [==============================] - 7s 109ms/step - loss: 0.1332 - accuracy: 0.9575\n",
            "Epoch 6/75\n",
            "64/64 [==============================] - 8s 120ms/step - loss: 0.1076 - accuracy: 0.9651\n",
            "Epoch 7/75\n",
            "64/64 [==============================] - 6s 94ms/step - loss: 0.0757 - accuracy: 0.9796\n",
            "Epoch 8/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0681 - accuracy: 0.9794\n",
            "Epoch 9/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0453 - accuracy: 0.9887\n",
            "Epoch 10/75\n",
            "64/64 [==============================] - 5s 79ms/step - loss: 0.0574 - accuracy: 0.9816\n",
            "Epoch 11/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0445 - accuracy: 0.9882\n",
            "Epoch 12/75\n",
            "64/64 [==============================] - 6s 93ms/step - loss: 0.0294 - accuracy: 0.9907\n",
            "Epoch 13/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0206 - accuracy: 0.9948\n",
            "Epoch 14/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0238 - accuracy: 0.9944\n",
            "Epoch 15/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.0117 - accuracy: 0.9968\n",
            "Epoch 16/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0172 - accuracy: 0.9966\n",
            "Epoch 17/75\n",
            "64/64 [==============================] - 6s 92ms/step - loss: 0.0127 - accuracy: 0.9963\n",
            "Epoch 18/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0149 - accuracy: 0.9958\n",
            "Epoch 19/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0282 - accuracy: 0.9907\n",
            "Epoch 20/75\n",
            "64/64 [==============================] - 5s 77ms/step - loss: 0.0169 - accuracy: 0.9956\n",
            "Epoch 21/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0170 - accuracy: 0.9944\n",
            "Epoch 22/75\n",
            "64/64 [==============================] - 6s 92ms/step - loss: 0.0179 - accuracy: 0.9953\n",
            "Epoch 23/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0134 - accuracy: 0.9975\n",
            "Epoch 24/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0132 - accuracy: 0.9956\n",
            "Epoch 25/75\n",
            "64/64 [==============================] - 5s 78ms/step - loss: 0.0141 - accuracy: 0.9961\n",
            "Epoch 26/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0125 - accuracy: 0.9973\n",
            "Epoch 27/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0052 - accuracy: 0.9978\n",
            "Epoch 28/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0063 - accuracy: 0.9975\n",
            "Epoch 29/75\n",
            "64/64 [==============================] - 5s 85ms/step - loss: 0.0102 - accuracy: 0.9978\n",
            "Epoch 30/75\n",
            "64/64 [==============================] - 5s 79ms/step - loss: 0.0062 - accuracy: 0.9988\n",
            "Epoch 31/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0065 - accuracy: 0.9990\n",
            "Epoch 32/75\n",
            "64/64 [==============================] - 6s 92ms/step - loss: 0.0140 - accuracy: 0.9966\n",
            "Epoch 33/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0108 - accuracy: 0.9975\n",
            "Epoch 34/75\n",
            "64/64 [==============================] - 5s 84ms/step - loss: 0.0050 - accuracy: 0.9988\n",
            "Epoch 35/75\n",
            "64/64 [==============================] - 5s 79ms/step - loss: 0.0077 - accuracy: 0.9975\n",
            "Epoch 36/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0063 - accuracy: 0.9988\n",
            "Epoch 37/75\n",
            "64/64 [==============================] - 6s 92ms/step - loss: 0.0078 - accuracy: 0.9980\n",
            "Epoch 38/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0157 - accuracy: 0.9958\n",
            "Epoch 39/75\n",
            "64/64 [==============================] - 5s 83ms/step - loss: 0.0845 - accuracy: 0.9759\n",
            "Epoch 40/75\n",
            "64/64 [==============================] - 5s 80ms/step - loss: 0.0278 - accuracy: 0.9929\n",
            "Epoch 41/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0087 - accuracy: 0.9983\n",
            "Epoch 42/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0050 - accuracy: 0.9988\n",
            "Epoch 43/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0052 - accuracy: 0.9993\n",
            "Epoch 44/75\n",
            "64/64 [==============================] - 5s 81ms/step - loss: 0.0045 - accuracy: 0.9990\n",
            "Epoch 45/75\n",
            "64/64 [==============================] - 5s 81ms/step - loss: 0.0040 - accuracy: 0.9988\n",
            "Epoch 46/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0029 - accuracy: 0.9995\n",
            "Epoch 47/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0030 - accuracy: 0.9995\n",
            "Epoch 48/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0045 - accuracy: 0.9993\n",
            "Epoch 49/75\n",
            "64/64 [==============================] - 5s 78ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 50/75\n",
            "64/64 [==============================] - 5s 84ms/step - loss: 0.0041 - accuracy: 0.9995\n",
            "Epoch 51/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0070 - accuracy: 0.9990\n",
            "Epoch 52/75\n",
            "64/64 [==============================] - 6s 93ms/step - loss: 0.0030 - accuracy: 0.9995\n",
            "Epoch 53/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0045 - accuracy: 0.9990\n",
            "Epoch 54/75\n",
            "64/64 [==============================] - 5s 78ms/step - loss: 0.0040 - accuracy: 0.9995\n",
            "Epoch 55/75\n",
            "64/64 [==============================] - 6s 85ms/step - loss: 0.0037 - accuracy: 0.9993\n",
            "Epoch 56/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0043 - accuracy: 0.9993\n",
            "Epoch 57/75\n",
            "64/64 [==============================] - 6s 90ms/step - loss: 0.0029 - accuracy: 0.9990\n",
            "Epoch 58/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0031 - accuracy: 0.9995\n",
            "Epoch 59/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0032 - accuracy: 0.9993\n",
            "Epoch 60/75\n",
            "64/64 [==============================] - 6s 87ms/step - loss: 0.0034 - accuracy: 0.9995\n",
            "Epoch 61/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0026 - accuracy: 0.9995\n",
            "Epoch 62/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0041 - accuracy: 0.9988\n",
            "Epoch 63/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0044 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "64/64 [==============================] - 5s 74ms/step - loss: 0.0047 - accuracy: 0.9995\n",
            "Epoch 65/75\n",
            "64/64 [==============================] - 6s 88ms/step - loss: 0.0043 - accuracy: 0.9988\n",
            "Epoch 66/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0047 - accuracy: 0.9995\n",
            "Epoch 67/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0026 - accuracy: 0.9993\n",
            "Epoch 68/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0022 - accuracy: 0.9995\n",
            "Epoch 69/75\n",
            "64/64 [==============================] - 5s 73ms/step - loss: 0.0022 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0031 - accuracy: 0.9995\n",
            "Epoch 71/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0029 - accuracy: 0.9993\n",
            "Epoch 72/75\n",
            "64/64 [==============================] - 6s 91ms/step - loss: 0.0024 - accuracy: 0.9998\n",
            "Epoch 73/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0022 - accuracy: 0.9998\n",
            "Epoch 74/75\n",
            "64/64 [==============================] - 5s 72ms/step - loss: 0.0020 - accuracy: 0.9995\n",
            "Epoch 75/75\n",
            "64/64 [==============================] - 6s 90ms/step - loss: 0.0019 - accuracy: 0.9998\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23bb27ec20>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qGVIl5pc7I6I",
        "outputId": "e3d79f84-e8d4-4783-8731-3e8f917439e8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "32/32 [==============================] - 0s 9ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BUMvaFEb7KxU",
        "outputId": "6689ebec-2cd7-468c-ffc8-8b6afddf0661"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9685658153241651,\n",
              " 1.0,\n",
              " 0.9402985074626866,\n",
              " 0.9371643725695945,\n",
              " 0.9390199798525514)"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bZ-Z_zMs7Mzz",
        "outputId": "92c8c55d-a5aa-4fe1-8968-71ae05111792"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9377431906614786"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "K17TeJXF7Omm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "9D_Kgu1u7SQI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "d8xnsAFe7WDA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "llkhRaJa7X7w"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "BHcatHvL7Zuw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "JRtyHy0t7bhQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "bHifzTTw7dZI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "JrbEXYR47fRJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "WTTfxVTj7g_4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "1kcaCKOF7iVw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "Rx0NJh5g7j-8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "w_H4C_c07liY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6SEzc9dA7nKk",
        "outputId": "7e31eeb2-4768-45f2-e24b-9204398af169"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 7s 83ms/step - loss: 0.4659 - accuracy: 0.7899\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.3670 - accuracy: 0.8497\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.2933 - accuracy: 0.8858\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.2385 - accuracy: 0.9107\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.1775 - accuracy: 0.9334\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.1315 - accuracy: 0.9533\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0848 - accuracy: 0.9717\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0720 - accuracy: 0.9765\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0635 - accuracy: 0.9799\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.0456 - accuracy: 0.9857\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0598 - accuracy: 0.9794\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0302 - accuracy: 0.9908\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0274 - accuracy: 0.9908\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0242 - accuracy: 0.9937\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0200 - accuracy: 0.9954\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0170 - accuracy: 0.9949\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0136 - accuracy: 0.9952\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0162 - accuracy: 0.9964\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0147 - accuracy: 0.9964\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0433 - accuracy: 0.9879\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0323 - accuracy: 0.9903\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0127 - accuracy: 0.9969\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0089 - accuracy: 0.9988\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0116 - accuracy: 0.9973\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0067 - accuracy: 0.9983\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0068 - accuracy: 0.9983\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0064 - accuracy: 0.9981\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0092 - accuracy: 0.9981\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0056 - accuracy: 0.9988\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0062 - accuracy: 0.9990\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0056 - accuracy: 0.9988\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0098 - accuracy: 0.9976\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.0084 - accuracy: 0.9981\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0056 - accuracy: 0.9983\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.0096 - accuracy: 0.9969\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0063 - accuracy: 0.9990\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0076 - accuracy: 0.9978\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.0389 - accuracy: 0.9879\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0253 - accuracy: 0.9920\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 6s 88ms/step - loss: 0.0165 - accuracy: 0.9949\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0127 - accuracy: 0.9971\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0091 - accuracy: 0.9985\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0097 - accuracy: 0.9976\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0073 - accuracy: 0.9976\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.0054 - accuracy: 0.9981\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0165 - accuracy: 0.9961\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0126 - accuracy: 0.9959\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0146 - accuracy: 0.9959\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0085 - accuracy: 0.9966\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.0079 - accuracy: 0.9988\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0048 - accuracy: 0.9988\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0051 - accuracy: 0.9988\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0041 - accuracy: 0.9988\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0034 - accuracy: 0.9993\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 6s 88ms/step - loss: 0.0038 - accuracy: 0.9988\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0033 - accuracy: 0.9990\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0052 - accuracy: 0.9983\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0057 - accuracy: 0.9985\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0043 - accuracy: 0.9990\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.0064 - accuracy: 0.9988\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0038 - accuracy: 0.9993\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0043 - accuracy: 0.9993\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0039 - accuracy: 0.9990\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0054 - accuracy: 0.9990\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.0029 - accuracy: 0.9993\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0064 - accuracy: 0.9985\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0051 - accuracy: 0.9993\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0037 - accuracy: 0.9993\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0040 - accuracy: 0.9985\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 6s 86ms/step - loss: 0.0038 - accuracy: 0.9993\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0039 - accuracy: 0.9990\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0052 - accuracy: 0.9983\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 6s 90ms/step - loss: 0.0185 - accuracy: 0.9947\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0710 - accuracy: 0.9792\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 8s 124ms/step - loss: 0.0213 - accuracy: 0.9939\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23bb0adcc0>"
            ]
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_QNdUgaH7o04",
        "outputId": "c7c34f68-62bf-43fb-a5d7-3c722e8ff036"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 8ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zFo2zlET7qxx",
        "outputId": "b87ef3e5-9476-42af-ddcc-ccb7d93b89a3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9825581395348837,\n",
              " 0.9960707269155207,\n",
              " 0.9694072657743786,\n",
              " 0.9651226976586996,\n",
              " 0.9654779926898992)"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5Vb3hs7O7s82",
        "outputId": "bd69629a-1cbc-4d1e-ccdf-003fbeaf9423"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9694072657743786"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "Z-LUgpjD7u6o"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "_5B7RVcm7w7R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "SMt2Lmaa70Jx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "5yvU_Khy713A"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "WXEwQwTY73S0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "kZxzKIOw747v"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "4y4Af8ZX76lA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "6vyOwZFT78QA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "9EYnMjQ97-GB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "egbeQTfo7_uf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "OZQJ1AGO8BIw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "UuzZmlcS8CmA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FmdU2Wj68EoZ",
        "outputId": "d8002f88-2308-44c5-f902-cace0f49b670"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 2s 70ms/step - loss: 0.6479 - accuracy: 0.6027\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.5338 - accuracy: 0.7471\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.4595 - accuracy: 0.7888\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 68ms/step - loss: 0.4753 - accuracy: 0.7703\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.4233 - accuracy: 0.8014\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.4047 - accuracy: 0.8159\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 2s 100ms/step - loss: 0.3771 - accuracy: 0.8362\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 2s 110ms/step - loss: 0.3503 - accuracy: 0.8391\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.3510 - accuracy: 0.8450\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.3420 - accuracy: 0.8459\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.3028 - accuracy: 0.8702\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.2573 - accuracy: 0.8915\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.2566 - accuracy: 0.8837\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.2490 - accuracy: 0.8963\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.2105 - accuracy: 0.9118\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.1897 - accuracy: 0.9225\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 2s 112ms/step - loss: 0.1835 - accuracy: 0.9254\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 2s 96ms/step - loss: 0.2020 - accuracy: 0.9176\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.1571 - accuracy: 0.9380\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.1245 - accuracy: 0.9496\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.1084 - accuracy: 0.9661\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.1243 - accuracy: 0.9554\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0887 - accuracy: 0.9671\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0679 - accuracy: 0.9767\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0537 - accuracy: 0.9864\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0438 - accuracy: 0.9913\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 2s 121ms/step - loss: 0.0395 - accuracy: 0.9942\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 84ms/step - loss: 0.0446 - accuracy: 0.9874\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.1586 - accuracy: 0.9612\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0900 - accuracy: 0.9748\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0587 - accuracy: 0.9835\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0318 - accuracy: 0.9961\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0268 - accuracy: 0.9952\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0192 - accuracy: 0.9952\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.0906 - accuracy: 0.9651\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 89ms/step - loss: 0.1992 - accuracy: 0.9234\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 2s 142ms/step - loss: 0.1008 - accuracy: 0.9612\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 2s 110ms/step - loss: 0.0570 - accuracy: 0.9835\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 2s 102ms/step - loss: 0.0343 - accuracy: 0.9884\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 2s 118ms/step - loss: 0.0224 - accuracy: 0.9961\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 2s 98ms/step - loss: 0.0160 - accuracy: 0.9961\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 2s 95ms/step - loss: 0.0159 - accuracy: 0.9971\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 2s 127ms/step - loss: 0.0158 - accuracy: 0.9942\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 3s 150ms/step - loss: 0.0172 - accuracy: 0.9952\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 2s 105ms/step - loss: 0.0298 - accuracy: 0.9913\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 2s 107ms/step - loss: 0.0194 - accuracy: 0.9932\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 2s 110ms/step - loss: 0.0144 - accuracy: 0.9961\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 2s 129ms/step - loss: 0.0159 - accuracy: 0.9942\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 2s 132ms/step - loss: 0.0140 - accuracy: 0.9961\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 3s 152ms/step - loss: 0.0117 - accuracy: 0.9971\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 2s 112ms/step - loss: 0.0160 - accuracy: 0.9961\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 2s 90ms/step - loss: 0.0188 - accuracy: 0.9942\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0165 - accuracy: 0.9952\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0153 - accuracy: 0.9961\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0072 - accuracy: 0.9981\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0299 - accuracy: 0.9903\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0463 - accuracy: 0.9884\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 2s 116ms/step - loss: 0.0780 - accuracy: 0.9797\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 2s 95ms/step - loss: 0.0345 - accuracy: 0.9884\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.0247 - accuracy: 0.9922\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0300 - accuracy: 0.9893\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0223 - accuracy: 0.9952\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0164 - accuracy: 0.9942\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0236 - accuracy: 0.9932\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0577 - accuracy: 0.9884\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.0143 - accuracy: 0.9971\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 75ms/step - loss: 0.0149 - accuracy: 0.9961\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 2s 119ms/step - loss: 0.0080 - accuracy: 0.9990\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 1s 81ms/step - loss: 0.0081 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 69ms/step - loss: 0.0069 - accuracy: 0.9990\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 71ms/step - loss: 0.0167 - accuracy: 0.9942\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.0098 - accuracy: 0.9971\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 2s 113ms/step - loss: 0.0104 - accuracy: 0.9971\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 2s 111ms/step - loss: 0.0096 - accuracy: 0.9981\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 2s 123ms/step - loss: 0.0071 - accuracy: 0.9990\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b9565870>"
            ]
          },
          "metadata": {},
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iXT9mEXT8GY_",
        "outputId": "65edb9c4-9574-42a5-e3c4-320e24a1563e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 13ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Zm7td-WO8IQv",
        "outputId": "5c1d8c05-82ee-46a6-ee15-1a095534abbd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9573643410852714,\n",
              " 1.0,\n",
              " 0.9266666666666666,\n",
              " 0.9136388533868907,\n",
              " 0.9170651031204625)"
            ]
          },
          "metadata": {},
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2Mmup2tr8KtQ",
        "outputId": "20af7feb-5f21-4c68-d9ec-62371ed7f263"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.907563025210084"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CNN(Geary)**"
      ],
      "metadata": {
        "id": "5_ShZ0_OAHLu"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "6sTHgfxO7wb1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "sqRv-EUhANJN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "U0EUPPiJAVyA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "U5zVU0oXAcL9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "FSuyCnE9Adys"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "L6QbxYVCAhzd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "yyxBR6LQAjxg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "ulOWoYw4AmAF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "TCSkd-ZpAnxO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "QOIt6IcpApe4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "HLaaUNmxArit"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vjqi9jWUAtYt",
        "outputId": "2af3f04e-f833-401a-cc27-2a758d9945ba"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 6s 80ms/step - loss: 0.5119 - accuracy: 0.8056\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.4968 - accuracy: 0.8056\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 5s 114ms/step - loss: 0.4820 - accuracy: 0.8056\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 4s 88ms/step - loss: 0.4766 - accuracy: 0.8052\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 3s 61ms/step - loss: 0.4633 - accuracy: 0.8029\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 3s 62ms/step - loss: 0.4454 - accuracy: 0.8075\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.4437 - accuracy: 0.8125\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.4376 - accuracy: 0.8172\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 4s 85ms/step - loss: 0.4355 - accuracy: 0.8207\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 2s 57ms/step - loss: 0.4139 - accuracy: 0.8253\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 3s 78ms/step - loss: 0.4106 - accuracy: 0.8327\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.3919 - accuracy: 0.8373\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 2s 58ms/step - loss: 0.4024 - accuracy: 0.8385\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 2s 59ms/step - loss: 0.3770 - accuracy: 0.8466\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 2s 58ms/step - loss: 0.3670 - accuracy: 0.8540\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.3445 - accuracy: 0.8594\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 0.3419 - accuracy: 0.8567\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 4s 100ms/step - loss: 0.3127 - accuracy: 0.8772\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.3024 - accuracy: 0.8850\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.3022 - accuracy: 0.8838\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 4s 94ms/step - loss: 0.2783 - accuracy: 0.8962\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.2586 - accuracy: 0.8993\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.2531 - accuracy: 0.9074\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.2261 - accuracy: 0.9152\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 4s 104ms/step - loss: 0.2181 - accuracy: 0.9202\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 5s 116ms/step - loss: 0.2033 - accuracy: 0.9272\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.1963 - accuracy: 0.9303\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 4s 96ms/step - loss: 0.1853 - accuracy: 0.9307\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 4s 96ms/step - loss: 0.1764 - accuracy: 0.9330\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 4s 102ms/step - loss: 0.1587 - accuracy: 0.9361\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 4s 88ms/step - loss: 0.1472 - accuracy: 0.9419\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.1573 - accuracy: 0.9361\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 5s 114ms/step - loss: 0.1302 - accuracy: 0.9524\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 0.1254 - accuracy: 0.9566\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.1009 - accuracy: 0.9628\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 5s 117ms/step - loss: 0.0894 - accuracy: 0.9694\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.1094 - accuracy: 0.9566\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 3s 82ms/step - loss: 0.0874 - accuracy: 0.9667\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 6s 140ms/step - loss: 0.0653 - accuracy: 0.9779\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 5s 120ms/step - loss: 0.0710 - accuracy: 0.9737\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 4s 88ms/step - loss: 0.0586 - accuracy: 0.9810\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 5s 112ms/step - loss: 0.0506 - accuracy: 0.9837\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 4s 101ms/step - loss: 0.0534 - accuracy: 0.9841\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.0522 - accuracy: 0.9826\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 5s 111ms/step - loss: 0.0552 - accuracy: 0.9802\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 4s 88ms/step - loss: 0.0547 - accuracy: 0.9818\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.0446 - accuracy: 0.9872\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 4s 97ms/step - loss: 0.0363 - accuracy: 0.9903\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 5s 118ms/step - loss: 0.0410 - accuracy: 0.9849\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.0386 - accuracy: 0.9880\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.0309 - accuracy: 0.9903\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 5s 110ms/step - loss: 0.0330 - accuracy: 0.9915\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 4s 90ms/step - loss: 0.0283 - accuracy: 0.9930\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.0236 - accuracy: 0.9938\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 5s 113ms/step - loss: 0.0221 - accuracy: 0.9954\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 3s 85ms/step - loss: 0.0250 - accuracy: 0.9934\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 4s 86ms/step - loss: 0.0151 - accuracy: 0.9977\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 5s 123ms/step - loss: 0.0117 - accuracy: 0.9973\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 5s 114ms/step - loss: 0.0087 - accuracy: 0.9981\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 4s 99ms/step - loss: 0.0066 - accuracy: 0.9985\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 4s 109ms/step - loss: 0.0065 - accuracy: 0.9988\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 4s 91ms/step - loss: 0.0070 - accuracy: 0.9981\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0049 - accuracy: 0.9988\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 2s 59ms/step - loss: 0.0064 - accuracy: 0.9988\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 3s 68ms/step - loss: 0.0062 - accuracy: 0.9981\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.0045 - accuracy: 0.9988\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 2s 58ms/step - loss: 0.0058 - accuracy: 0.9985\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 3s 64ms/step - loss: 0.0069 - accuracy: 0.9981\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 3s 81ms/step - loss: 0.0066 - accuracy: 0.9985\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 4s 108ms/step - loss: 0.0083 - accuracy: 0.9988\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 4s 87ms/step - loss: 0.0353 - accuracy: 0.9888\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 4s 89ms/step - loss: 0.1997 - accuracy: 0.9291\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 4s 92ms/step - loss: 0.1446 - accuracy: 0.9423\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 5s 111ms/step - loss: 0.0384 - accuracy: 0.9907\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 5s 115ms/step - loss: 0.0175 - accuracy: 0.9981\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b931ebc0>"
            ]
          },
          "metadata": {},
          "execution_count": 73
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fh-Tg6XyAvNU",
        "outputId": "4a6a2312-0b22-4122-b74d-9f5afdff3e92"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 1s 18ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C6rp-lJ-Axv0",
        "outputId": "7e772edd-d434-422f-a969-270c0aa501a7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9395348837209302,\n",
              " 0.9370629370629371,\n",
              " 0.8170731707317073,\n",
              " 0.8335329658330631,\n",
              " 0.8369018445633083)"
            ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n5tHcwzMA0pk",
        "outputId": "b66e524c-4a76-400f-c1e9-8e4d461d4354"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9402390438247012"
            ]
          },
          "metadata": {},
          "execution_count": 76
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "epFBRivN71d-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "4RDJyY0m73Kq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "_hiQuFfS79kN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "-r0POUiu7_jz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "rdntTFfo8B1s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "iJj3YRz38D_K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "Sjn-xRNs8HIC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "LDMHOT098JrB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "yhY0ivQc8MAK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "sh5XQFPi8O0x"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "KYZSqeL_8RQ5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "as0Me22T8Tpx",
        "outputId": "dd727c89-b1c8-48c4-990a-b8ec7cb5fa66"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 3s 52ms/step - loss: 0.5062 - accuracy: 0.8087\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 2s 59ms/step - loss: 0.4929 - accuracy: 0.8087\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.4918 - accuracy: 0.8087\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 2s 61ms/step - loss: 0.4804 - accuracy: 0.8087\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.4643 - accuracy: 0.8096\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 2s 57ms/step - loss: 0.4509 - accuracy: 0.8131\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 2s 56ms/step - loss: 0.4460 - accuracy: 0.8162\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.4399 - accuracy: 0.8144\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 3s 71ms/step - loss: 0.4310 - accuracy: 0.8251\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.4271 - accuracy: 0.8189\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.4133 - accuracy: 0.8330\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.4233 - accuracy: 0.8268\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 2s 57ms/step - loss: 0.3965 - accuracy: 0.8397\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.3983 - accuracy: 0.8485\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 3s 72ms/step - loss: 0.3876 - accuracy: 0.8459\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 2s 58ms/step - loss: 0.3836 - accuracy: 0.8561\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 2s 56ms/step - loss: 0.3594 - accuracy: 0.8521\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 2s 57ms/step - loss: 0.3496 - accuracy: 0.8649\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.3455 - accuracy: 0.8636\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 3s 93ms/step - loss: 0.3272 - accuracy: 0.8733\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.3248 - accuracy: 0.8756\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 2s 67ms/step - loss: 0.3085 - accuracy: 0.8840\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 2s 69ms/step - loss: 0.3011 - accuracy: 0.8857\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 2s 64ms/step - loss: 0.3064 - accuracy: 0.8840\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 3s 74ms/step - loss: 0.2837 - accuracy: 0.8902\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.2761 - accuracy: 0.8973\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.2488 - accuracy: 0.9110\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.2402 - accuracy: 0.9163\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 2s 57ms/step - loss: 0.2356 - accuracy: 0.9163\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 2s 60ms/step - loss: 0.2183 - accuracy: 0.9252\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 2s 65ms/step - loss: 0.2227 - accuracy: 0.9154\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1920 - accuracy: 0.9318\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.2064 - accuracy: 0.9172\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1891 - accuracy: 0.9296\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1600 - accuracy: 0.9424\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1591 - accuracy: 0.9371\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 2s 50ms/step - loss: 0.1537 - accuracy: 0.9442\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 2s 54ms/step - loss: 0.1493 - accuracy: 0.9442\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 2s 44ms/step - loss: 0.1343 - accuracy: 0.9491\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1194 - accuracy: 0.9562\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1104 - accuracy: 0.9646\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.1716 - accuracy: 0.9345\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.1442 - accuracy: 0.9429\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0951 - accuracy: 0.9672\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0807 - accuracy: 0.9712\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 2s 58ms/step - loss: 0.0706 - accuracy: 0.9725\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0757 - accuracy: 0.9752\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0629 - accuracy: 0.9783\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0573 - accuracy: 0.9827\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0582 - accuracy: 0.9810\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0664 - accuracy: 0.9765\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0941 - accuracy: 0.9628\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.0607 - accuracy: 0.9779\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 2s 55ms/step - loss: 0.0767 - accuracy: 0.9734\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0705 - accuracy: 0.9765\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0440 - accuracy: 0.9872\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0337 - accuracy: 0.9911\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0283 - accuracy: 0.9916\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0260 - accuracy: 0.9934\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.0252 - accuracy: 0.9916\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 2s 44ms/step - loss: 0.0223 - accuracy: 0.9956\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 2s 53ms/step - loss: 0.0449 - accuracy: 0.9845\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 1s 39ms/step - loss: 0.0293 - accuracy: 0.9903\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0236 - accuracy: 0.9929\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0263 - accuracy: 0.9920\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0162 - accuracy: 0.9947\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0227 - accuracy: 0.9925\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0185 - accuracy: 0.9942\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 2s 49ms/step - loss: 0.0289 - accuracy: 0.9911\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 2s 42ms/step - loss: 0.0219 - accuracy: 0.9934\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0285 - accuracy: 0.9903\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0387 - accuracy: 0.9907\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 1s 41ms/step - loss: 0.0262 - accuracy: 0.9929\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0510 - accuracy: 0.9818\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 1s 40ms/step - loss: 0.0703 - accuracy: 0.9739\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x795a8c99ba00>"
            ]
          },
          "metadata": {},
          "execution_count": 97
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "awLjMr388WLB",
        "outputId": "5722bb9f-33fe-4063-e7a1-a1f70ab9ddc7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Hn9JFDRF8Yzc",
        "outputId": "98a89a95-e3a9-4aa8-fee7-f70610711826"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9091847265221878,\n",
              " 0.7551020408163265,\n",
              " 0.8685446009389671,\n",
              " 0.8078602620087335,\n",
              " 0.7487803153503501,\n",
              " 0.751924405996307)"
            ]
          },
          "metadata": {},
          "execution_count": 99
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iBpxg-1O8cQ0",
        "outputId": "22a3cb1d-0322-42fd-d8db-af488224c477"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9206349206349206"
            ]
          },
          "metadata": {},
          "execution_count": 100
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "W4viTL2SA9lp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "PVwyMfo6A7zt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "t9SXmnCXBGWN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "xlAB9qr2BKvl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "5AZPhEPHBM-A"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "j6spvM9vBPit"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "y2xnKMD4BRrl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "mrkKKXPTBWQ1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "olaQTIHcBYMV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "jQyFi8OqBZzt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "e0iHtrPmBbbk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "C4DeRW2XBc6B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3jJFTQCEBevd",
        "outputId": "98b3c270-c194-4627-acd3-efa9952a47c5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 8s 91ms/step - loss: 0.6936 - accuracy: 0.5086\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 7s 102ms/step - loss: 0.6810 - accuracy: 0.5512\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.6514 - accuracy: 0.6295\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 7s 112ms/step - loss: 0.6292 - accuracy: 0.6643\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 6s 84ms/step - loss: 0.5969 - accuracy: 0.6927\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 8s 116ms/step - loss: 0.5571 - accuracy: 0.7255\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 6s 96ms/step - loss: 0.5155 - accuracy: 0.7508\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 7s 107ms/step - loss: 0.4625 - accuracy: 0.7869\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 6s 96ms/step - loss: 0.4016 - accuracy: 0.8204\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 6s 94ms/step - loss: 0.3577 - accuracy: 0.8429\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 6s 87ms/step - loss: 0.2890 - accuracy: 0.8855\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 8s 116ms/step - loss: 0.2615 - accuracy: 0.8942\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 7s 109ms/step - loss: 0.2121 - accuracy: 0.9207\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 7s 115ms/step - loss: 0.1697 - accuracy: 0.9407\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 6s 98ms/step - loss: 0.1536 - accuracy: 0.9458\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 7s 113ms/step - loss: 0.1199 - accuracy: 0.9622\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 6s 89ms/step - loss: 0.0968 - accuracy: 0.9684\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 7s 103ms/step - loss: 0.0750 - accuracy: 0.9773\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 7s 104ms/step - loss: 0.0718 - accuracy: 0.9783\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 9s 132ms/step - loss: 0.0544 - accuracy: 0.9855\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 8s 120ms/step - loss: 0.0602 - accuracy: 0.9814\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 7s 106ms/step - loss: 0.0388 - accuracy: 0.9908\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 8s 122ms/step - loss: 0.0330 - accuracy: 0.9899\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 6s 98ms/step - loss: 0.0300 - accuracy: 0.9930\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 7s 114ms/step - loss: 0.0249 - accuracy: 0.9949\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 6s 99ms/step - loss: 0.0342 - accuracy: 0.9896\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 8s 121ms/step - loss: 0.0324 - accuracy: 0.9901\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 6s 91ms/step - loss: 0.0210 - accuracy: 0.9947\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 7s 108ms/step - loss: 0.0127 - accuracy: 0.9978\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0067 - accuracy: 0.9986\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0054 - accuracy: 0.9993\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0043 - accuracy: 0.9990\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0043 - accuracy: 0.9990\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0038 - accuracy: 0.9993\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 5s 79ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0036 - accuracy: 0.9988\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0037 - accuracy: 0.9988\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0033 - accuracy: 0.9993\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0029 - accuracy: 0.9993\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0028 - accuracy: 0.9990\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0028 - accuracy: 0.9990\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0034 - accuracy: 0.9990\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0027 - accuracy: 0.9990\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0027 - accuracy: 0.9993\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0028 - accuracy: 0.9993\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0027 - accuracy: 0.9990\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.1309 - accuracy: 0.9619\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.1367 - accuracy: 0.9523\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0274 - accuracy: 0.9928\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0157 - accuracy: 0.9952\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0083 - accuracy: 0.9986\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.0040 - accuracy: 0.9990\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0031 - accuracy: 0.9993\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0029 - accuracy: 0.9988\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0027 - accuracy: 0.9993\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0023 - accuracy: 0.9995\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0023 - accuracy: 0.9993\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0022 - accuracy: 0.9993\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0019 - accuracy: 0.9995\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0019 - accuracy: 0.9988\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0018 - accuracy: 0.9995\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0019 - accuracy: 0.9993\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0018 - accuracy: 0.9995\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0017 - accuracy: 0.9993\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0030 - accuracy: 0.9993\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0020 - accuracy: 0.9995\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0018 - accuracy: 0.9995\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0020 - accuracy: 0.9993\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0024 - accuracy: 0.9988\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0019 - accuracy: 0.9990\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0018 - accuracy: 0.9993\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0015 - accuracy: 0.9995\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0015 - accuracy: 0.9995\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 6s 88ms/step - loss: 0.0016 - accuracy: 0.9993\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b3fc5f90>"
            ]
          },
          "metadata": {},
          "execution_count": 88
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qzHm6LNdBguX",
        "outputId": "b7ada317-8a19-4c66-f072-41df662c3aac"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 9ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ge3aQsKWBiy8",
        "outputId": "c9653215-bb8d-42fb-e3f3-0256eeba159f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9778206364513018,\n",
              " 0.9963369963369964,\n",
              " 0.9628318584070796,\n",
              " 0.955429271929841,\n",
              " 0.9560775500668928)"
            ]
          },
          "metadata": {},
          "execution_count": 90
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aBxPBgecBkp8",
        "outputId": "9c73286c-b2f5-469c-bf95-735a98e1e6fe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9572301425661914"
            ]
          },
          "metadata": {},
          "execution_count": 91
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "U9i3VjrvBtYE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "StyScHglBv8E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "LcNWDoNIBz2Q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "LuH1iVukB1is"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "W1rydyyJB3Rl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "JYqWuSoVB5Js"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "xJmLLaBgB6rR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "_G5P84b3B8S4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "2aJbVGW6B-Jc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "cos1qcjYB_rd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "3585Sc7_CBUw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "qYf1Ef6BCC5t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dsi8jgJPCEns",
        "outputId": "21905acc-2c77-4343-ef5b-8d0d5c1277df"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 6s 61ms/step - loss: 0.6871 - accuracy: 0.5319\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.6541 - accuracy: 0.6203\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.6256 - accuracy: 0.6670\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.5922 - accuracy: 0.6924\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.5591 - accuracy: 0.7137\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.4958 - accuracy: 0.7616\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.4501 - accuracy: 0.7916\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 4s 59ms/step - loss: 0.3886 - accuracy: 0.8221\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.3314 - accuracy: 0.8623\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.2693 - accuracy: 0.8909\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.2208 - accuracy: 0.9167\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 6s 100ms/step - loss: 0.1787 - accuracy: 0.9395\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 6s 94ms/step - loss: 0.1624 - accuracy: 0.9419\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.1209 - accuracy: 0.9606\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.1054 - accuracy: 0.9654\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0892 - accuracy: 0.9700\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 5s 77ms/step - loss: 0.0700 - accuracy: 0.9782\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0546 - accuracy: 0.9840\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0593 - accuracy: 0.9809\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 5s 80ms/step - loss: 0.0586 - accuracy: 0.9811\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0335 - accuracy: 0.9918\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 7s 108ms/step - loss: 0.0298 - accuracy: 0.9923\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 6s 97ms/step - loss: 0.0301 - accuracy: 0.9918\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0266 - accuracy: 0.9935\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 5s 78ms/step - loss: 0.0241 - accuracy: 0.9927\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0295 - accuracy: 0.9923\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 6s 88ms/step - loss: 0.0175 - accuracy: 0.9966\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 6s 95ms/step - loss: 0.0084 - accuracy: 0.9981\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0154 - accuracy: 0.9973\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0103 - accuracy: 0.9981\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 5s 72ms/step - loss: 0.0119 - accuracy: 0.9969\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0340 - accuracy: 0.9889\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0749 - accuracy: 0.9739\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0450 - accuracy: 0.9823\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0264 - accuracy: 0.9913\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0191 - accuracy: 0.9935\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 5s 73ms/step - loss: 0.0235 - accuracy: 0.9923\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0105 - accuracy: 0.9976\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.0071 - accuracy: 0.9988\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 5s 74ms/step - loss: 0.0062 - accuracy: 0.9988\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0048 - accuracy: 0.9983\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0055 - accuracy: 0.9985\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 5s 76ms/step - loss: 0.0054 - accuracy: 0.9983\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0068 - accuracy: 0.9983\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0056 - accuracy: 0.9985\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 5s 75ms/step - loss: 0.0056 - accuracy: 0.9988\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0097 - accuracy: 0.9976\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0047 - accuracy: 0.9990\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0071 - accuracy: 0.9983\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0067 - accuracy: 0.9981\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0073 - accuracy: 0.9983\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0075 - accuracy: 0.9985\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0196 - accuracy: 0.9944\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.1354 - accuracy: 0.9608\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0641 - accuracy: 0.9775\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0111 - accuracy: 0.9981\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 4s 64ms/step - loss: 0.0055 - accuracy: 0.9988\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 8s 123ms/step - loss: 0.0058 - accuracy: 0.9985\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 6s 100ms/step - loss: 0.0047 - accuracy: 0.9993\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 4s 66ms/step - loss: 0.0084 - accuracy: 0.9978\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0045 - accuracy: 0.9988\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0042 - accuracy: 0.9983\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0048 - accuracy: 0.9983\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0051 - accuracy: 0.9990\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 5s 71ms/step - loss: 0.0034 - accuracy: 0.9993\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 4s 67ms/step - loss: 0.0033 - accuracy: 0.9990\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 4s 61ms/step - loss: 0.0035 - accuracy: 0.9990\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0043 - accuracy: 0.9993\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0032 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.0043 - accuracy: 0.9988\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0030 - accuracy: 0.9990\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 4s 67ms/step - loss: 0.0056 - accuracy: 0.9988\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 4s 62ms/step - loss: 0.0036 - accuracy: 0.9988\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 5s 70ms/step - loss: 0.0047 - accuracy: 0.9983\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 4s 68ms/step - loss: 0.0046 - accuracy: 0.9993\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b4703fd0>"
            ]
          },
          "metadata": {},
          "execution_count": 103
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jPl_mzJiCGV3",
        "outputId": "80952ef9-97f2-443c-b9dd-3cc4e8d18ae9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Jg9vOlYiCIZ8",
        "outputId": "fbf10436-fc21-41e2-82ab-9f001604cda6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9573643410852714,\n",
              " 0.9781312127236581,\n",
              " 0.9371428571428572,\n",
              " 0.9147661362074985,\n",
              " 0.9155979699107222)"
            ]
          },
          "metadata": {},
          "execution_count": 105
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4QaByz7ICLDG",
        "outputId": "4d3730d8-8c81-468f-d936-448aadf598da"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9376181474480151"
            ]
          },
          "metadata": {},
          "execution_count": 106
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "lLXn92LHCNJz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "RjoDX4fwCPef"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "y1bhEG6lCTx0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "Hr0_soZ9CV1-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "uWrxbwC1CXx0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "vkU4rpPyCZcH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "sghSvMDiCbME"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=256, kernel_size=3, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))"
      ],
      "metadata": {
        "id": "vhLnwGUfCcy1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "ZV_b2_aqCeaN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "g4RKqpgiCgUv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "jWH0K11RChwt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "WuHTy484Cjlu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0w3SjdBMCldV",
        "outputId": "5227f539-e261-4b06-91a5-4697159777d1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 3s 58ms/step - loss: 0.6896 - accuracy: 0.5291\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.6585 - accuracy: 0.5959\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.6328 - accuracy: 0.6240\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 56ms/step - loss: 0.5560 - accuracy: 0.7151\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.5343 - accuracy: 0.7248\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4797 - accuracy: 0.7713\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.4864 - accuracy: 0.7645\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 2s 94ms/step - loss: 0.4620 - accuracy: 0.7829\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 75ms/step - loss: 0.4400 - accuracy: 0.8014\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.4278 - accuracy: 0.8081\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4250 - accuracy: 0.8101\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4351 - accuracy: 0.7926\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4448 - accuracy: 0.7926\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.4158 - accuracy: 0.8072\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4061 - accuracy: 0.8169\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4003 - accuracy: 0.8207\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.4218 - accuracy: 0.7965\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4157 - accuracy: 0.8043\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 72ms/step - loss: 0.3822 - accuracy: 0.8362\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 2s 95ms/step - loss: 0.3688 - accuracy: 0.8450\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 75ms/step - loss: 0.4213 - accuracy: 0.8004\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3749 - accuracy: 0.8372\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.4097 - accuracy: 0.8052\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.3589 - accuracy: 0.8556\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.3667 - accuracy: 0.8391\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.3381 - accuracy: 0.8634\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.3334 - accuracy: 0.8547\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.3373 - accuracy: 0.8566\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.3466 - accuracy: 0.8479\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3185 - accuracy: 0.8595\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 77ms/step - loss: 0.3052 - accuracy: 0.8721\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 2s 93ms/step - loss: 0.3011 - accuracy: 0.8750\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 70ms/step - loss: 0.2934 - accuracy: 0.8779\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.3045 - accuracy: 0.8692\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.2786 - accuracy: 0.8915\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.2721 - accuracy: 0.8886\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 1s 60ms/step - loss: 0.2544 - accuracy: 0.9021\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.2619 - accuracy: 0.8818\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.2425 - accuracy: 0.8983\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.2318 - accuracy: 0.9099\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.2260 - accuracy: 0.9099\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.2233 - accuracy: 0.9157\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 81ms/step - loss: 0.2142 - accuracy: 0.9157\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 2s 93ms/step - loss: 0.2610 - accuracy: 0.8837\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 65ms/step - loss: 0.2533 - accuracy: 0.8983\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.2063 - accuracy: 0.9186\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.2409 - accuracy: 0.9060\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 63ms/step - loss: 0.2209 - accuracy: 0.9128\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1829 - accuracy: 0.9302\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1696 - accuracy: 0.9351\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1819 - accuracy: 0.9302\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1651 - accuracy: 0.9370\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1581 - accuracy: 0.9380\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.2003 - accuracy: 0.9167\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 2s 91ms/step - loss: 0.1437 - accuracy: 0.9545\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 2s 96ms/step - loss: 0.1450 - accuracy: 0.9477\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.1247 - accuracy: 0.9583\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1232 - accuracy: 0.9632\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1465 - accuracy: 0.9457\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.1100 - accuracy: 0.9612\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.0971 - accuracy: 0.9690\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.1113 - accuracy: 0.9583\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.1547 - accuracy: 0.9390\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.1206 - accuracy: 0.9583\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.0944 - accuracy: 0.9700\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.0785 - accuracy: 0.9787\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 2s 96ms/step - loss: 0.0757 - accuracy: 0.9767\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 1s 87ms/step - loss: 0.0702 - accuracy: 0.9806\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.0706 - accuracy: 0.9787\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.0841 - accuracy: 0.9748\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0714 - accuracy: 0.9787\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.0632 - accuracy: 0.9816\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0604 - accuracy: 0.9835\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.0473 - accuracy: 0.9884\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 1s 58ms/step - loss: 0.0422 - accuracy: 0.9913\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b4833310>"
            ]
          },
          "metadata": {},
          "execution_count": 118
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1FFnJF42CnLL",
        "outputId": "a1d97a86-8ed5-4b2b-c3dc-e77e64112cfe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MoM8gofACpGU",
        "outputId": "99feb824-01c1-4ee2-b144-18fe01e6394d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9224806201550387,\n",
              " 0.916030534351145,\n",
              " 0.9302325581395349,\n",
              " 0.8449612403100775,\n",
              " 0.8450628103597665)"
            ]
          },
          "metadata": {},
          "execution_count": 120
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xaIwkgLlCq7s",
        "outputId": "64631570-086a-4019-f949-44a5d7c7bf30"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9291338582677166"
            ]
          },
          "metadata": {},
          "execution_count": 121
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **ANN(APAAC)**"
      ],
      "metadata": {
        "id": "AbE79UmWNZlG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "3MSb8q3BNojM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "bXfY_8CtNfWV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "LoPznkvUNrWe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "p1Ak834lNwBf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "bFJ_BoxvNyTl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "BUjaohh9N0_N"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "dYKVBWPzN6Ll"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "UAh78inyN86l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "WzsCOidYN_L2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "p7dFJu-WOBTU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "_jpHA-T9ODg9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VXn-hucaOFlc",
        "outputId": "68a7ff39-6fe5-4a36-ff8e-649bb5078aa8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 6s 55ms/step - loss: 0.4839 - accuracy: 0.7839\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 2s 43ms/step - loss: 0.4494 - accuracy: 0.8071\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 2s 50ms/step - loss: 0.4207 - accuracy: 0.8156\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.3973 - accuracy: 0.8265\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 2s 56ms/step - loss: 0.3789 - accuracy: 0.8381\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.3693 - accuracy: 0.8435\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 2s 56ms/step - loss: 0.3533 - accuracy: 0.8447\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.3442 - accuracy: 0.8482\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 2s 49ms/step - loss: 0.3394 - accuracy: 0.8505\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.3242 - accuracy: 0.8610\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.3196 - accuracy: 0.8621\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 3s 83ms/step - loss: 0.3083 - accuracy: 0.8753\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 3s 62ms/step - loss: 0.3062 - accuracy: 0.8722\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 2s 57ms/step - loss: 0.3048 - accuracy: 0.8807\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 2s 58ms/step - loss: 0.2942 - accuracy: 0.8796\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 3s 70ms/step - loss: 0.2852 - accuracy: 0.8854\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 3s 71ms/step - loss: 0.2864 - accuracy: 0.8792\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 2s 49ms/step - loss: 0.2775 - accuracy: 0.8900\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 0.2782 - accuracy: 0.8846\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 2s 49ms/step - loss: 0.2648 - accuracy: 0.8931\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.2586 - accuracy: 0.8927\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 2s 57ms/step - loss: 0.2544 - accuracy: 0.9001\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 2s 60ms/step - loss: 0.2550 - accuracy: 0.8931\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.2330 - accuracy: 0.9040\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.2273 - accuracy: 0.9063\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.2264 - accuracy: 0.9032\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 2s 44ms/step - loss: 0.2058 - accuracy: 0.9136\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 2s 45ms/step - loss: 0.2036 - accuracy: 0.9101\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 3s 84ms/step - loss: 0.1974 - accuracy: 0.9222\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.1896 - accuracy: 0.9202\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.1760 - accuracy: 0.9253\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.1737 - accuracy: 0.9291\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.1903 - accuracy: 0.9198\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.1610 - accuracy: 0.9369\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.1542 - accuracy: 0.9376\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.1458 - accuracy: 0.9438\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 2s 49ms/step - loss: 0.1467 - accuracy: 0.9361\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 2s 57ms/step - loss: 0.1384 - accuracy: 0.9423\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 2s 50ms/step - loss: 0.1276 - accuracy: 0.9504\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 2s 58ms/step - loss: 0.1281 - accuracy: 0.9477\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 3s 72ms/step - loss: 0.1217 - accuracy: 0.9531\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 2s 59ms/step - loss: 0.1154 - accuracy: 0.9551\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.1059 - accuracy: 0.9578\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 2s 52ms/step - loss: 0.1119 - accuracy: 0.9531\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 2s 50ms/step - loss: 0.1001 - accuracy: 0.9609\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 2s 52ms/step - loss: 0.1001 - accuracy: 0.9624\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 3s 69ms/step - loss: 0.1146 - accuracy: 0.9547\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0961 - accuracy: 0.9663\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 2s 50ms/step - loss: 0.0808 - accuracy: 0.9671\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0653 - accuracy: 0.9791\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.0762 - accuracy: 0.9760\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0722 - accuracy: 0.9729\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 2s 57ms/step - loss: 0.0595 - accuracy: 0.9775\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 2s 56ms/step - loss: 0.0955 - accuracy: 0.9593\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0770 - accuracy: 0.9667\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 2s 47ms/step - loss: 0.0701 - accuracy: 0.9717\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.0689 - accuracy: 0.9737\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.0540 - accuracy: 0.9775\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.0714 - accuracy: 0.9710\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 3s 66ms/step - loss: 0.0533 - accuracy: 0.9810\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 0.0438 - accuracy: 0.9853\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0588 - accuracy: 0.9791\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 2s 55ms/step - loss: 0.0537 - accuracy: 0.9826\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 2s 56ms/step - loss: 0.0402 - accuracy: 0.9884\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 3s 63ms/step - loss: 0.0351 - accuracy: 0.9899\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 3s 75ms/step - loss: 0.0309 - accuracy: 0.9911\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.0308 - accuracy: 0.9915\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 2s 53ms/step - loss: 0.0235 - accuracy: 0.9946\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 2s 57ms/step - loss: 0.0209 - accuracy: 0.9969\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 2s 43ms/step - loss: 0.0176 - accuracy: 0.9973\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 2s 51ms/step - loss: 0.0215 - accuracy: 0.9942\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 3s 61ms/step - loss: 0.0190 - accuracy: 0.9957\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 2s 48ms/step - loss: 0.0530 - accuracy: 0.9837\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 2s 46ms/step - loss: 0.0598 - accuracy: 0.9826\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 2s 41ms/step - loss: 0.0501 - accuracy: 0.9818\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b40e48b0>"
            ]
          },
          "metadata": {},
          "execution_count": 132
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wATu2BVnOIAc",
        "outputId": "8be329b1-7878-4f14-a12a-d54e0f6019d8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 0s 11ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Yf_jRY7BOLA7",
        "outputId": "357a7ddc-41ef-48a7-c511-d6499d5846d8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9565891472868217,\n",
              " 0.9296875,\n",
              " 0.8623188405797102,\n",
              " 0.867441758048179,\n",
              " 0.8684154879857565)"
            ]
          },
          "metadata": {},
          "execution_count": 134
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7ss8TG_gOODo",
        "outputId": "87e9ec23-5121-42a0-bd53-2a6d48781c06"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9632495164410058"
            ]
          },
          "metadata": {},
          "execution_count": 135
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "gSKWaPSFkAnp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "XjMI464dkClt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "U7p5TwxekIl2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "t2wWCmNukK7i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "CmOt1T9AkNX2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "ZCBA6Q6PkRJe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(n_timesteps,n_features)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "jWw-wh1fkT0m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "9EMXFYe3kdKu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "e0yZKWfykfhm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "49Xxq6ikkhoO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "UIjSc6Fskj2t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tW58RBUykmMe",
        "outputId": "40d5f47c-1e3c-4fb6-8108-1d9968ae5e80"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 2s 26ms/step - loss: 0.4744 - accuracy: 0.8091\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 1s 30ms/step - loss: 0.4351 - accuracy: 0.8162\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 1s 34ms/step - loss: 0.4233 - accuracy: 0.8167\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 1s 35ms/step - loss: 0.4102 - accuracy: 0.8220\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.3925 - accuracy: 0.8242\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.3741 - accuracy: 0.8397\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.3647 - accuracy: 0.8485\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.3508 - accuracy: 0.8521\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.3406 - accuracy: 0.8583\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.3290 - accuracy: 0.8663\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.3220 - accuracy: 0.8716\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.3166 - accuracy: 0.8667\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.3051 - accuracy: 0.8791\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.3000 - accuracy: 0.8791\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 1s 33ms/step - loss: 0.2891 - accuracy: 0.8884\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 1s 32ms/step - loss: 0.2849 - accuracy: 0.8862\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.2768 - accuracy: 0.8871\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.2808 - accuracy: 0.8862\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.2773 - accuracy: 0.8866\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.2577 - accuracy: 0.8902\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.2469 - accuracy: 0.8995\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 1s 29ms/step - loss: 0.2619 - accuracy: 0.8862\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 1s 21ms/step - loss: 0.2438 - accuracy: 0.8946\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.2327 - accuracy: 0.8999\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.2198 - accuracy: 0.9092\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.2091 - accuracy: 0.9150\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.2113 - accuracy: 0.9083\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.2034 - accuracy: 0.9145\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 1s 24ms/step - loss: 0.1818 - accuracy: 0.9238\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.1738 - accuracy: 0.9296\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 1s 21ms/step - loss: 0.1879 - accuracy: 0.9167\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1699 - accuracy: 0.9318\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.1615 - accuracy: 0.9274\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1444 - accuracy: 0.9389\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1401 - accuracy: 0.9411\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1400 - accuracy: 0.9411\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1223 - accuracy: 0.9508\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1132 - accuracy: 0.9557\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1286 - accuracy: 0.9446\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0969 - accuracy: 0.9641\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0863 - accuracy: 0.9672\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0915 - accuracy: 0.9655\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0775 - accuracy: 0.9717\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0765 - accuracy: 0.9690\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.1028 - accuracy: 0.9593\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0894 - accuracy: 0.9641\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.0563 - accuracy: 0.9787\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.0594 - accuracy: 0.9787\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0577 - accuracy: 0.9841\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0534 - accuracy: 0.9832\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0406 - accuracy: 0.9863\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.0408 - accuracy: 0.9889\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0726 - accuracy: 0.9748\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0898 - accuracy: 0.9641\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0782 - accuracy: 0.9717\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0450 - accuracy: 0.9867\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0455 - accuracy: 0.9858\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.0340 - accuracy: 0.9907\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 1s 21ms/step - loss: 0.0261 - accuracy: 0.9938\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 1s 22ms/step - loss: 0.0198 - accuracy: 0.9965\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 1s 25ms/step - loss: 0.0209 - accuracy: 0.9956\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0307 - accuracy: 0.9894\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 1s 23ms/step - loss: 0.0552 - accuracy: 0.9787\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.0765 - accuracy: 0.9748\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 1s 21ms/step - loss: 0.0504 - accuracy: 0.9801\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0288 - accuracy: 0.9942\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0411 - accuracy: 0.9872\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0346 - accuracy: 0.9867\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0259 - accuracy: 0.9911\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0174 - accuracy: 0.9960\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0119 - accuracy: 0.9973\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0116 - accuracy: 0.9973\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.0126 - accuracy: 0.9978\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0117 - accuracy: 0.9973\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.0372 - accuracy: 0.9903\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x795a9107d9f0>"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MYh-9kZPkpQ6",
        "outputId": "f0072396-9a7f-47bb-957c-6cca3e929ecc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 7ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3lK2jlh5ksDC",
        "outputId": "be0e20e9-cf8e-4b6e-9cb2-73e45d137782"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9195046439628483,\n",
              " 0.8157894736842105,\n",
              " 0.8378378378378378,\n",
              " 0.8266666666666665,\n",
              " 0.7742594484167518,\n",
              " 0.7743762065669784)"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mKlXxfz8kw3o",
        "outputId": "3093a1fb-ae79-4047-8e5d-4378516d8fe6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9437751004016064"
            ]
          },
          "metadata": {},
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "83vA6rg8OVD6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "Qgp5o-NhOZy1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "JEX5pFrdOeAl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "uEYZzIkKOirV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "nYV23S98OmMN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "lk28fHrWOoev"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "ylkbRXNjOqnB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "0X-Vn3qYOsWO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "5TABKaLLOvJ9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "cDXaThNOOxUn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "f-bM2ygCOzDt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "FDYX_J-SO2Ft"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XyBIiBX0O4Pt",
        "outputId": "1d856f7a-ad6e-4893-c534-afa18acb38a5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "64/64 [==============================] - 6s 57ms/step - loss: 0.6241 - accuracy: 0.6466\n",
            "Epoch 2/75\n",
            "64/64 [==============================] - 3s 50ms/step - loss: 0.5389 - accuracy: 0.7252\n",
            "Epoch 3/75\n",
            "64/64 [==============================] - 3s 44ms/step - loss: 0.4496 - accuracy: 0.7763\n",
            "Epoch 4/75\n",
            "64/64 [==============================] - 3s 53ms/step - loss: 0.3909 - accuracy: 0.8111\n",
            "Epoch 5/75\n",
            "64/64 [==============================] - 4s 58ms/step - loss: 0.3557 - accuracy: 0.8274\n",
            "Epoch 6/75\n",
            "64/64 [==============================] - 3s 53ms/step - loss: 0.3483 - accuracy: 0.8357\n",
            "Epoch 7/75\n",
            "64/64 [==============================] - 3s 52ms/step - loss: 0.2952 - accuracy: 0.8637\n",
            "Epoch 8/75\n",
            "64/64 [==============================] - 3s 51ms/step - loss: 0.2689 - accuracy: 0.8816\n",
            "Epoch 9/75\n",
            "64/64 [==============================] - 4s 69ms/step - loss: 0.2339 - accuracy: 0.9035\n",
            "Epoch 10/75\n",
            "64/64 [==============================] - 4s 66ms/step - loss: 0.2157 - accuracy: 0.9131\n",
            "Epoch 11/75\n",
            "64/64 [==============================] - 4s 61ms/step - loss: 0.1985 - accuracy: 0.9209\n",
            "Epoch 12/75\n",
            "64/64 [==============================] - 4s 63ms/step - loss: 0.1690 - accuracy: 0.9386\n",
            "Epoch 13/75\n",
            "64/64 [==============================] - 4s 55ms/step - loss: 0.1566 - accuracy: 0.9366\n",
            "Epoch 14/75\n",
            "64/64 [==============================] - 3s 41ms/step - loss: 0.1410 - accuracy: 0.9489\n",
            "Epoch 15/75\n",
            "64/64 [==============================] - 3s 41ms/step - loss: 0.1189 - accuracy: 0.9556\n",
            "Epoch 16/75\n",
            "64/64 [==============================] - 3s 46ms/step - loss: 0.1278 - accuracy: 0.9497\n",
            "Epoch 17/75\n",
            "64/64 [==============================] - 3s 48ms/step - loss: 0.1051 - accuracy: 0.9575\n",
            "Epoch 18/75\n",
            "64/64 [==============================] - 2s 25ms/step - loss: 0.0869 - accuracy: 0.9686\n",
            "Epoch 19/75\n",
            "64/64 [==============================] - 2s 24ms/step - loss: 0.0844 - accuracy: 0.9713\n",
            "Epoch 20/75\n",
            "64/64 [==============================] - 2s 24ms/step - loss: 0.0712 - accuracy: 0.9759\n",
            "Epoch 21/75\n",
            "64/64 [==============================] - 2s 25ms/step - loss: 0.0801 - accuracy: 0.9703\n",
            "Epoch 22/75\n",
            "64/64 [==============================] - 2s 25ms/step - loss: 0.0564 - accuracy: 0.9811\n",
            "Epoch 23/75\n",
            "64/64 [==============================] - 3s 39ms/step - loss: 0.0661 - accuracy: 0.9750\n",
            "Epoch 24/75\n",
            "64/64 [==============================] - 3s 49ms/step - loss: 0.0625 - accuracy: 0.9799\n",
            "Epoch 25/75\n",
            "64/64 [==============================] - 2s 34ms/step - loss: 0.0433 - accuracy: 0.9870\n",
            "Epoch 26/75\n",
            "64/64 [==============================] - 2s 39ms/step - loss: 0.0435 - accuracy: 0.9882\n",
            "Epoch 27/75\n",
            "64/64 [==============================] - 2s 36ms/step - loss: 0.0874 - accuracy: 0.9693\n",
            "Epoch 28/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0733 - accuracy: 0.9747\n",
            "Epoch 29/75\n",
            "64/64 [==============================] - 4s 68ms/step - loss: 0.0307 - accuracy: 0.9926\n",
            "Epoch 30/75\n",
            "64/64 [==============================] - 3s 51ms/step - loss: 0.0362 - accuracy: 0.9902\n",
            "Epoch 31/75\n",
            "64/64 [==============================] - 3s 49ms/step - loss: 0.0427 - accuracy: 0.9870\n",
            "Epoch 32/75\n",
            "64/64 [==============================] - 4s 61ms/step - loss: 0.0455 - accuracy: 0.9848\n",
            "Epoch 33/75\n",
            "64/64 [==============================] - 3s 49ms/step - loss: 0.0447 - accuracy: 0.9850\n",
            "Epoch 34/75\n",
            "64/64 [==============================] - 3s 47ms/step - loss: 0.0248 - accuracy: 0.9926\n",
            "Epoch 35/75\n",
            "64/64 [==============================] - 3s 42ms/step - loss: 0.0228 - accuracy: 0.9936\n",
            "Epoch 36/75\n",
            "64/64 [==============================] - 3s 47ms/step - loss: 0.0326 - accuracy: 0.9894\n",
            "Epoch 37/75\n",
            "64/64 [==============================] - 4s 55ms/step - loss: 0.0321 - accuracy: 0.9885\n",
            "Epoch 38/75\n",
            "64/64 [==============================] - 3s 40ms/step - loss: 0.0229 - accuracy: 0.9909\n",
            "Epoch 39/75\n",
            "64/64 [==============================] - 2s 38ms/step - loss: 0.0175 - accuracy: 0.9956\n",
            "Epoch 40/75\n",
            "64/64 [==============================] - 3s 43ms/step - loss: 0.0220 - accuracy: 0.9948\n",
            "Epoch 41/75\n",
            "64/64 [==============================] - 3s 52ms/step - loss: 0.0232 - accuracy: 0.9934\n",
            "Epoch 42/75\n",
            "64/64 [==============================] - 3s 48ms/step - loss: 0.0199 - accuracy: 0.9946\n",
            "Epoch 43/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0145 - accuracy: 0.9966\n",
            "Epoch 44/75\n",
            "64/64 [==============================] - 3s 51ms/step - loss: 0.0195 - accuracy: 0.9934\n",
            "Epoch 45/75\n",
            "64/64 [==============================] - 4s 63ms/step - loss: 0.0115 - accuracy: 0.9971\n",
            "Epoch 46/75\n",
            "64/64 [==============================] - 4s 70ms/step - loss: 0.0323 - accuracy: 0.9894\n",
            "Epoch 47/75\n",
            "64/64 [==============================] - 3s 50ms/step - loss: 0.0352 - accuracy: 0.9867\n",
            "Epoch 48/75\n",
            "64/64 [==============================] - 3s 44ms/step - loss: 0.0269 - accuracy: 0.9924\n",
            "Epoch 49/75\n",
            "64/64 [==============================] - 3s 54ms/step - loss: 0.0434 - accuracy: 0.9862\n",
            "Epoch 50/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0243 - accuracy: 0.9929\n",
            "Epoch 51/75\n",
            "64/64 [==============================] - 3s 47ms/step - loss: 0.0437 - accuracy: 0.9867\n",
            "Epoch 52/75\n",
            "64/64 [==============================] - 3s 42ms/step - loss: 0.0357 - accuracy: 0.9875\n",
            "Epoch 53/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0187 - accuracy: 0.9956\n",
            "Epoch 54/75\n",
            "64/64 [==============================] - 3s 53ms/step - loss: 0.0160 - accuracy: 0.9966\n",
            "Epoch 55/75\n",
            "64/64 [==============================] - 3s 43ms/step - loss: 0.0106 - accuracy: 0.9971\n",
            "Epoch 56/75\n",
            "64/64 [==============================] - 3s 40ms/step - loss: 0.0100 - accuracy: 0.9973\n",
            "Epoch 57/75\n",
            "64/64 [==============================] - 3s 43ms/step - loss: 0.0120 - accuracy: 0.9966\n",
            "Epoch 58/75\n",
            "64/64 [==============================] - 3s 53ms/step - loss: 0.0121 - accuracy: 0.9966\n",
            "Epoch 59/75\n",
            "64/64 [==============================] - 3s 48ms/step - loss: 0.0081 - accuracy: 0.9983\n",
            "Epoch 60/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0128 - accuracy: 0.9958\n",
            "Epoch 61/75\n",
            "64/64 [==============================] - 3s 53ms/step - loss: 0.0099 - accuracy: 0.9980\n",
            "Epoch 62/75\n",
            "64/64 [==============================] - 4s 64ms/step - loss: 0.0084 - accuracy: 0.9975\n",
            "Epoch 63/75\n",
            "64/64 [==============================] - 4s 55ms/step - loss: 0.0098 - accuracy: 0.9988\n",
            "Epoch 64/75\n",
            "64/64 [==============================] - 3s 49ms/step - loss: 0.0121 - accuracy: 0.9973\n",
            "Epoch 65/75\n",
            "64/64 [==============================] - 3s 46ms/step - loss: 0.0077 - accuracy: 0.9985\n",
            "Epoch 66/75\n",
            "64/64 [==============================] - 4s 57ms/step - loss: 0.0050 - accuracy: 0.9995\n",
            "Epoch 67/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0039 - accuracy: 0.9995\n",
            "Epoch 68/75\n",
            "64/64 [==============================] - 3s 45ms/step - loss: 0.0039 - accuracy: 0.9995\n",
            "Epoch 69/75\n",
            "64/64 [==============================] - 3s 41ms/step - loss: 0.0033 - accuracy: 0.9990\n",
            "Epoch 70/75\n",
            "64/64 [==============================] - 3s 46ms/step - loss: 0.0123 - accuracy: 0.9961\n",
            "Epoch 71/75\n",
            "64/64 [==============================] - 4s 55ms/step - loss: 0.0751 - accuracy: 0.9777\n",
            "Epoch 72/75\n",
            "64/64 [==============================] - 3s 50ms/step - loss: 0.0350 - accuracy: 0.9889\n",
            "Epoch 73/75\n",
            "64/64 [==============================] - 4s 55ms/step - loss: 0.0178 - accuracy: 0.9946\n",
            "Epoch 74/75\n",
            "64/64 [==============================] - 4s 61ms/step - loss: 0.0091 - accuracy: 0.9983\n",
            "Epoch 75/75\n",
            "64/64 [==============================] - 4s 54ms/step - loss: 0.0062 - accuracy: 0.9990\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b0e20520>"
            ]
          },
          "metadata": {},
          "execution_count": 147
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "STD4VkpyO66V",
        "outputId": "884bd24e-6187-4428-c704-09ec2d87db9a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "32/32 [==============================] - 1s 9ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yzT-3bgnO9x9",
        "outputId": "403291e1-185c-4c9c-eed0-142d03948a5b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9675834970530451,\n",
              " 0.994,\n",
              " 0.9430740037950665,\n",
              " 0.9352075080137169,\n",
              " 0.9365243799779303)"
            ]
          },
          "metadata": {},
          "execution_count": 149
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gmnya6O-PAYM",
        "outputId": "169e3aca-73e1-4626-9c2f-b7d95b9615b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9420849420849421"
            ]
          },
          "metadata": {},
          "execution_count": 150
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "JrIGYi18PLNJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "XOlc_1plPNu5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "S00LMJwlPRlw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "OHuVLFtTPUGM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "wp8BShw5PXA2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "AO-W60BdPZDX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "y6p2yjICPbL8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "z5TAr6zPPdp8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "_JOAWscxPfpE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "_rC9HabOPhx-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "gQ1nl20VPkDN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "gGSi1Zg2PmBH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_UX-sgfXPoFY",
        "outputId": "64a27c3e-c204-429b-a1b5-74137eba2f40"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 7s 50ms/step - loss: 0.5822 - accuracy: 0.6883\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.4822 - accuracy: 0.7737\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 3s 46ms/step - loss: 0.4385 - accuracy: 0.8011\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 4s 57ms/step - loss: 0.4052 - accuracy: 0.8134\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 3s 50ms/step - loss: 0.3737 - accuracy: 0.8318\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.3401 - accuracy: 0.8509\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 3s 47ms/step - loss: 0.3198 - accuracy: 0.8642\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.2944 - accuracy: 0.8717\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 4s 59ms/step - loss: 0.2622 - accuracy: 0.8872\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 0.2597 - accuracy: 0.8865\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.2385 - accuracy: 0.8904\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 4s 60ms/step - loss: 0.2091 - accuracy: 0.9119\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 4s 63ms/step - loss: 0.1977 - accuracy: 0.9172\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 3s 47ms/step - loss: 0.1756 - accuracy: 0.9252\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 3s 49ms/step - loss: 0.1618 - accuracy: 0.9344\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 4s 57ms/step - loss: 0.1359 - accuracy: 0.9453\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 3s 48ms/step - loss: 0.1687 - accuracy: 0.9320\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.1209 - accuracy: 0.9538\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.0942 - accuracy: 0.9683\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 3s 50ms/step - loss: 0.0867 - accuracy: 0.9719\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 3s 52ms/step - loss: 0.0994 - accuracy: 0.9637\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0774 - accuracy: 0.9760\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.0852 - accuracy: 0.9705\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0680 - accuracy: 0.9773\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 4s 65ms/step - loss: 0.0541 - accuracy: 0.9821\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.0718 - accuracy: 0.9734\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0572 - accuracy: 0.9823\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.0684 - accuracy: 0.9768\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0617 - accuracy: 0.9787\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 3s 47ms/step - loss: 0.0375 - accuracy: 0.9894\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0345 - accuracy: 0.9889\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0324 - accuracy: 0.9910\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 3s 45ms/step - loss: 0.0265 - accuracy: 0.9935\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0342 - accuracy: 0.9874\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 4s 55ms/step - loss: 0.0551 - accuracy: 0.9845\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 3s 46ms/step - loss: 0.0337 - accuracy: 0.9906\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 3s 45ms/step - loss: 0.0197 - accuracy: 0.9959\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 0.0134 - accuracy: 0.9964\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 0.0250 - accuracy: 0.9925\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 3s 46ms/step - loss: 0.0185 - accuracy: 0.9952\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0244 - accuracy: 0.9920\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 4s 54ms/step - loss: 0.0146 - accuracy: 0.9973\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 3s 48ms/step - loss: 0.0159 - accuracy: 0.9976\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.0141 - accuracy: 0.9971\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.0116 - accuracy: 0.9971\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0150 - accuracy: 0.9966\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0393 - accuracy: 0.9874\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0460 - accuracy: 0.9860\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0266 - accuracy: 0.9925\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0122 - accuracy: 0.9961\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0265 - accuracy: 0.9923\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0352 - accuracy: 0.9881\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.0504 - accuracy: 0.9816\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0374 - accuracy: 0.9872\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 2s 24ms/step - loss: 0.0344 - accuracy: 0.9879\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0125 - accuracy: 0.9971\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0171 - accuracy: 0.9952\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0076 - accuracy: 0.9981\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0113 - accuracy: 0.9976\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0069 - accuracy: 0.9983\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0117 - accuracy: 0.9966\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.0122 - accuracy: 0.9961\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0127 - accuracy: 0.9969\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0107 - accuracy: 0.9973\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0083 - accuracy: 0.9976\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0073 - accuracy: 0.9981\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0047 - accuracy: 0.9993\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.0133 - accuracy: 0.9964\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.0179 - accuracy: 0.9942\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0487 - accuracy: 0.9831\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0162 - accuracy: 0.9956\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.0171 - accuracy: 0.9944\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0265 - accuracy: 0.9910\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 2s 25ms/step - loss: 0.0195 - accuracy: 0.9949\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0137 - accuracy: 0.9949\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b029d840>"
            ]
          },
          "metadata": {},
          "execution_count": 162
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yYUKE7dLPp--",
        "outputId": "d23efaf7-e1ea-43d2-81b1-1479e88683a7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 1s 8ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IHY77yzgPsr8",
        "outputId": "858c4100-b6e3-4953-8edd-3f1d3267dcc6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9748062015503876,\n",
              " 0.9908592321755028,\n",
              " 0.9626998223801065,\n",
              " 0.9493351560995321,\n",
              " 0.9497969126095036)"
            ]
          },
          "metadata": {},
          "execution_count": 164
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2XCM86IWPvP_",
        "outputId": "fa52c1ff-dc7e-44af-ee49-b7b633507d04"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9567010309278351"
            ]
          },
          "metadata": {},
          "execution_count": 165
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "_l5BeCzDP2KB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')"
      ],
      "metadata": {
        "id": "DEPhgUz0P44X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "TPkKkKZTP778"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "4Vw3S0ibP-Xs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "gzRFBbqGQAf2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "ZSYF7itiQCUU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "OVvX3Db2QEFH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "CSCwaEc7QF8N"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "brzPyzNcQKOH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "F-EoW0DPQLK0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "HCSZYcMhQNes"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "YI7wuDNCQPqJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wdUn4JjVQSTk",
        "outputId": "f977aa11-438c-42cf-d757-c3161cd2d548"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 2s 24ms/step - loss: 0.6545 - accuracy: 0.5591\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.5762 - accuracy: 0.6841\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.5376 - accuracy: 0.7209\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.4998 - accuracy: 0.7297\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.4890 - accuracy: 0.7345\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.4827 - accuracy: 0.7548\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.4949 - accuracy: 0.7597\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.4533 - accuracy: 0.7800\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.4368 - accuracy: 0.7878\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.4398 - accuracy: 0.7946\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 0s 23ms/step - loss: 0.4187 - accuracy: 0.8062\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.4092 - accuracy: 0.8130\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.4028 - accuracy: 0.8207\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.3985 - accuracy: 0.8169\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.3985 - accuracy: 0.8178\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.3920 - accuracy: 0.8188\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 0s 23ms/step - loss: 0.4056 - accuracy: 0.8236\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.4096 - accuracy: 0.8159\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.3655 - accuracy: 0.8411\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 1s 39ms/step - loss: 0.3775 - accuracy: 0.8207\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 42ms/step - loss: 0.3992 - accuracy: 0.8033\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 42ms/step - loss: 0.3531 - accuracy: 0.8343\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.3326 - accuracy: 0.8566\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.3152 - accuracy: 0.8585\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.3186 - accuracy: 0.8527\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.3224 - accuracy: 0.8517\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.2979 - accuracy: 0.8614\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.3243 - accuracy: 0.8527\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.2851 - accuracy: 0.8769\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.2721 - accuracy: 0.8837\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.2817 - accuracy: 0.8779\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 0s 23ms/step - loss: 0.2501 - accuracy: 0.9002\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.2525 - accuracy: 0.8895\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 0s 23ms/step - loss: 0.2374 - accuracy: 0.8983\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.2580 - accuracy: 0.8944\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.3476 - accuracy: 0.8740\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.2408 - accuracy: 0.8934\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.2107 - accuracy: 0.9225\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 0s 26ms/step - loss: 0.2000 - accuracy: 0.9225\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1948 - accuracy: 0.9225\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1864 - accuracy: 0.9293\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.1898 - accuracy: 0.9254\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1559 - accuracy: 0.9438\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 0s 24ms/step - loss: 0.1693 - accuracy: 0.9399\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1659 - accuracy: 0.9448\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1357 - accuracy: 0.9496\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.1619 - accuracy: 0.9409\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.1405 - accuracy: 0.9457\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.1157 - accuracy: 0.9593\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 42ms/step - loss: 0.1454 - accuracy: 0.9448\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.1030 - accuracy: 0.9661\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.1078 - accuracy: 0.9700\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0794 - accuracy: 0.9797\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0932 - accuracy: 0.9709\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 59ms/step - loss: 0.0958 - accuracy: 0.9729\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.1602 - accuracy: 0.9341\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 48ms/step - loss: 0.1767 - accuracy: 0.9302\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.1144 - accuracy: 0.9622\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 42ms/step - loss: 0.0768 - accuracy: 0.9767\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 50ms/step - loss: 0.0937 - accuracy: 0.9690\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 56ms/step - loss: 0.0896 - accuracy: 0.9719\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 43ms/step - loss: 0.0920 - accuracy: 0.9719\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 63ms/step - loss: 0.0608 - accuracy: 0.9845\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 57ms/step - loss: 0.0467 - accuracy: 0.9893\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 47ms/step - loss: 0.0423 - accuracy: 0.9893\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 0s 28ms/step - loss: 0.0426 - accuracy: 0.9874\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0366 - accuracy: 0.9903\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0351 - accuracy: 0.9903\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0774 - accuracy: 0.9797\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0563 - accuracy: 0.9835\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0334 - accuracy: 0.9932\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1163 - accuracy: 0.9651\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0969 - accuracy: 0.9787\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.0663 - accuracy: 0.9816\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 0s 25ms/step - loss: 0.1891 - accuracy: 0.9419\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23b00b96c0>"
            ]
          },
          "metadata": {},
          "execution_count": 177
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8OPR7qs1QUPc",
        "outputId": "806b1f72-8a49-4575-c9ec-7b34f850b167"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3q3nAiXiQXY-",
        "outputId": "85b3bc1b-77a4-45b2-a305-3eb30c8883f1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9031007751937985,\n",
              " 0.9848484848484849,\n",
              " 0.8496732026143791,\n",
              " 0.8053594061198625,\n",
              " 0.8163444230951157)"
            ]
          },
          "metadata": {},
          "execution_count": 179
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uCNB151ZQaHU",
        "outputId": "6e05d3b8-f011-4c79-ebcb-e2404273da67"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8174603174603174"
            ]
          },
          "metadata": {},
          "execution_count": 180
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **ANN(Geary)**"
      ],
      "metadata": {
        "id": "nqcqt24FT2Hs"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "JPYJ_OmHT9ml"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "bBKehGcdT5sq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "8hLRhArYUBYM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "DckU-hqgUDhO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "PjP8BosLUFMx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "I3-LQSShUHqC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "r6zpBYonUJpT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "R5PCW3YEUMQC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "TUigXFALUONT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "gGe-ReyZUQ-y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "MPsGZ03GUT2b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j1dNVYrgUWCJ",
        "outputId": "2ab97072-2910-40b1-ce81-92cba2a0e64e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "41/41 [==============================] - 3s 22ms/step - loss: 0.5124 - accuracy: 0.8052\n",
            "Epoch 2/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4999 - accuracy: 0.8052\n",
            "Epoch 3/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4939 - accuracy: 0.8052\n",
            "Epoch 4/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.5011 - accuracy: 0.8052\n",
            "Epoch 5/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4904 - accuracy: 0.8052\n",
            "Epoch 6/75\n",
            "41/41 [==============================] - 1s 23ms/step - loss: 0.4900 - accuracy: 0.8052\n",
            "Epoch 7/75\n",
            "41/41 [==============================] - 1s 34ms/step - loss: 0.4842 - accuracy: 0.8052\n",
            "Epoch 8/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.4867 - accuracy: 0.8048\n",
            "Epoch 9/75\n",
            "41/41 [==============================] - 1s 29ms/step - loss: 0.4802 - accuracy: 0.8048\n",
            "Epoch 10/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4834 - accuracy: 0.8048\n",
            "Epoch 11/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4748 - accuracy: 0.8048\n",
            "Epoch 12/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4725 - accuracy: 0.8017\n",
            "Epoch 13/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4739 - accuracy: 0.8033\n",
            "Epoch 14/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4667 - accuracy: 0.8060\n",
            "Epoch 15/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4649 - accuracy: 0.8044\n",
            "Epoch 16/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4639 - accuracy: 0.8079\n",
            "Epoch 17/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4561 - accuracy: 0.8036\n",
            "Epoch 18/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4570 - accuracy: 0.8021\n",
            "Epoch 19/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4662 - accuracy: 0.8036\n",
            "Epoch 20/75\n",
            "41/41 [==============================] - 1s 24ms/step - loss: 0.4539 - accuracy: 0.8091\n",
            "Epoch 21/75\n",
            "41/41 [==============================] - 1s 35ms/step - loss: 0.4555 - accuracy: 0.8064\n",
            "Epoch 22/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.4520 - accuracy: 0.8060\n",
            "Epoch 23/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4504 - accuracy: 0.8036\n",
            "Epoch 24/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4553 - accuracy: 0.8009\n",
            "Epoch 25/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4474 - accuracy: 0.8110\n",
            "Epoch 26/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4465 - accuracy: 0.8048\n",
            "Epoch 27/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4431 - accuracy: 0.8064\n",
            "Epoch 28/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4387 - accuracy: 0.8098\n",
            "Epoch 29/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4421 - accuracy: 0.8079\n",
            "Epoch 30/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4388 - accuracy: 0.8095\n",
            "Epoch 31/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4307 - accuracy: 0.8137\n",
            "Epoch 32/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4295 - accuracy: 0.8122\n",
            "Epoch 33/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4258 - accuracy: 0.8199\n",
            "Epoch 34/75\n",
            "41/41 [==============================] - 1s 32ms/step - loss: 0.4349 - accuracy: 0.8125\n",
            "Epoch 35/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.4273 - accuracy: 0.8153\n",
            "Epoch 36/75\n",
            "41/41 [==============================] - 1s 29ms/step - loss: 0.4227 - accuracy: 0.8156\n",
            "Epoch 37/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4182 - accuracy: 0.8222\n",
            "Epoch 38/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4211 - accuracy: 0.8191\n",
            "Epoch 39/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4141 - accuracy: 0.8242\n",
            "Epoch 40/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4091 - accuracy: 0.8207\n",
            "Epoch 41/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.4044 - accuracy: 0.8261\n",
            "Epoch 42/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.4005 - accuracy: 0.8323\n",
            "Epoch 43/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3963 - accuracy: 0.8338\n",
            "Epoch 44/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3891 - accuracy: 0.8350\n",
            "Epoch 45/75\n",
            "41/41 [==============================] - 1s 23ms/step - loss: 0.4020 - accuracy: 0.8338\n",
            "Epoch 46/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3903 - accuracy: 0.8366\n",
            "Epoch 47/75\n",
            "41/41 [==============================] - 1s 27ms/step - loss: 0.3907 - accuracy: 0.8393\n",
            "Epoch 48/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.3799 - accuracy: 0.8385\n",
            "Epoch 49/75\n",
            "41/41 [==============================] - 1s 34ms/step - loss: 0.3681 - accuracy: 0.8443\n",
            "Epoch 50/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3633 - accuracy: 0.8513\n",
            "Epoch 51/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.3504 - accuracy: 0.8563\n",
            "Epoch 52/75\n",
            "41/41 [==============================] - 1s 23ms/step - loss: 0.3586 - accuracy: 0.8571\n",
            "Epoch 53/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3446 - accuracy: 0.8567\n",
            "Epoch 54/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.3446 - accuracy: 0.8590\n",
            "Epoch 55/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3295 - accuracy: 0.8652\n",
            "Epoch 56/75\n",
            "41/41 [==============================] - 1s 20ms/step - loss: 0.3388 - accuracy: 0.8528\n",
            "Epoch 57/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.3230 - accuracy: 0.8741\n",
            "Epoch 58/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.3169 - accuracy: 0.8699\n",
            "Epoch 59/75\n",
            "41/41 [==============================] - 1s 23ms/step - loss: 0.3124 - accuracy: 0.8768\n",
            "Epoch 60/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.3106 - accuracy: 0.8668\n",
            "Epoch 61/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.2970 - accuracy: 0.8799\n",
            "Epoch 62/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.2887 - accuracy: 0.8819\n",
            "Epoch 63/75\n",
            "41/41 [==============================] - 1s 25ms/step - loss: 0.2843 - accuracy: 0.8823\n",
            "Epoch 64/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.2765 - accuracy: 0.8877\n",
            "Epoch 65/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.2777 - accuracy: 0.8811\n",
            "Epoch 66/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.2779 - accuracy: 0.8896\n",
            "Epoch 67/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.2670 - accuracy: 0.8950\n",
            "Epoch 68/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.2575 - accuracy: 0.8958\n",
            "Epoch 69/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.2608 - accuracy: 0.8877\n",
            "Epoch 70/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.2350 - accuracy: 0.9016\n",
            "Epoch 71/75\n",
            "41/41 [==============================] - 1s 21ms/step - loss: 0.2333 - accuracy: 0.9059\n",
            "Epoch 72/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.2190 - accuracy: 0.9175\n",
            "Epoch 73/75\n",
            "41/41 [==============================] - 1s 22ms/step - loss: 0.2201 - accuracy: 0.9132\n",
            "Epoch 74/75\n",
            "41/41 [==============================] - 1s 30ms/step - loss: 0.2234 - accuracy: 0.9090\n",
            "Epoch 75/75\n",
            "41/41 [==============================] - 1s 36ms/step - loss: 0.2043 - accuracy: 0.9256\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x797f75333250>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fc_sl4kGUYEK",
        "outputId": "97772acc-86ad-48d8-fdf2-1b898cbc1c67"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21/21 [==============================] - 0s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4_EGnjeqUcJd",
        "outputId": "2d9428b3-c552-4a01-e13d-52f3c37640bd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8728682170542635,\n",
              " 0.6126760563380281,\n",
              " 0.7631578947368421,\n",
              " 0.6015639124932199,\n",
              " 0.6072098792162705)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "id": "3jTjmAZLUgyk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "91f181ca-b881-4a45-83b1-ad77493762f3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9463220675944334"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "qfrOU7_j9EZe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "dEWhw55v9F-p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "G5LUh-iK9PTM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "5mkEOGei9RY0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "BI71xh6-9UKb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "ynz8PPzo9Wkj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(n_timesteps,n_features)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "NP1oGQ6G9YUZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "gD8uXyte9avx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "iyzuNOJH9dOp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "pSafmCaT9fFy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "WVsXrM2I9hj6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(xtrain, ytrain, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OfVPHqu69jhf",
        "outputId": "06ad55a2-8aa5-4b3f-9472-798a935d78e2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "36/36 [==============================] - 2s 17ms/step - loss: 0.5097 - accuracy: 0.8087\n",
            "Epoch 2/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4944 - accuracy: 0.8087\n",
            "Epoch 3/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4955 - accuracy: 0.8087\n",
            "Epoch 4/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4940 - accuracy: 0.8087\n",
            "Epoch 5/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4916 - accuracy: 0.8087\n",
            "Epoch 6/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4889 - accuracy: 0.8087\n",
            "Epoch 7/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.4846 - accuracy: 0.8087\n",
            "Epoch 8/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4903 - accuracy: 0.8087\n",
            "Epoch 9/75\n",
            "36/36 [==============================] - 1s 25ms/step - loss: 0.4792 - accuracy: 0.8087\n",
            "Epoch 10/75\n",
            "36/36 [==============================] - 1s 23ms/step - loss: 0.4830 - accuracy: 0.8087\n",
            "Epoch 11/75\n",
            "36/36 [==============================] - 1s 22ms/step - loss: 0.4754 - accuracy: 0.8087\n",
            "Epoch 12/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4742 - accuracy: 0.8087\n",
            "Epoch 13/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4668 - accuracy: 0.8091\n",
            "Epoch 14/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4651 - accuracy: 0.8082\n",
            "Epoch 15/75\n",
            "36/36 [==============================] - 1s 22ms/step - loss: 0.4628 - accuracy: 0.8091\n",
            "Epoch 16/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4615 - accuracy: 0.8109\n",
            "Epoch 17/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4572 - accuracy: 0.8069\n",
            "Epoch 18/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4627 - accuracy: 0.8096\n",
            "Epoch 19/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4584 - accuracy: 0.8060\n",
            "Epoch 20/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4548 - accuracy: 0.8118\n",
            "Epoch 21/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4453 - accuracy: 0.8100\n",
            "Epoch 22/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4475 - accuracy: 0.8096\n",
            "Epoch 23/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4523 - accuracy: 0.8113\n",
            "Epoch 24/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.4423 - accuracy: 0.8136\n",
            "Epoch 25/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4424 - accuracy: 0.8131\n",
            "Epoch 26/75\n",
            "36/36 [==============================] - 1s 22ms/step - loss: 0.4493 - accuracy: 0.8127\n",
            "Epoch 27/75\n",
            "36/36 [==============================] - 1s 26ms/step - loss: 0.4443 - accuracy: 0.8082\n",
            "Epoch 28/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4416 - accuracy: 0.8109\n",
            "Epoch 29/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.4359 - accuracy: 0.8140\n",
            "Epoch 30/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4390 - accuracy: 0.8118\n",
            "Epoch 31/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.4354 - accuracy: 0.8136\n",
            "Epoch 32/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4352 - accuracy: 0.8122\n",
            "Epoch 33/75\n",
            "36/36 [==============================] - 1s 21ms/step - loss: 0.4294 - accuracy: 0.8171\n",
            "Epoch 34/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4356 - accuracy: 0.8162\n",
            "Epoch 35/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4346 - accuracy: 0.8149\n",
            "Epoch 36/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4286 - accuracy: 0.8175\n",
            "Epoch 37/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4300 - accuracy: 0.8206\n",
            "Epoch 38/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4255 - accuracy: 0.8202\n",
            "Epoch 39/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4226 - accuracy: 0.8233\n",
            "Epoch 40/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4220 - accuracy: 0.8229\n",
            "Epoch 41/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4238 - accuracy: 0.8224\n",
            "Epoch 42/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4264 - accuracy: 0.8211\n",
            "Epoch 43/75\n",
            "36/36 [==============================] - 1s 21ms/step - loss: 0.4231 - accuracy: 0.8268\n",
            "Epoch 44/75\n",
            "36/36 [==============================] - 1s 27ms/step - loss: 0.4165 - accuracy: 0.8282\n",
            "Epoch 45/75\n",
            "36/36 [==============================] - 1s 22ms/step - loss: 0.4168 - accuracy: 0.8255\n",
            "Epoch 46/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4192 - accuracy: 0.8224\n",
            "Epoch 47/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4232 - accuracy: 0.8255\n",
            "Epoch 48/75\n",
            "36/36 [==============================] - 1s 16ms/step - loss: 0.4098 - accuracy: 0.8260\n",
            "Epoch 49/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4102 - accuracy: 0.8317\n",
            "Epoch 50/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4064 - accuracy: 0.8353\n",
            "Epoch 51/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.4117 - accuracy: 0.8264\n",
            "Epoch 52/75\n",
            "36/36 [==============================] - 1s 16ms/step - loss: 0.4069 - accuracy: 0.8295\n",
            "Epoch 53/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4121 - accuracy: 0.8291\n",
            "Epoch 54/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.4006 - accuracy: 0.8366\n",
            "Epoch 55/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4016 - accuracy: 0.8348\n",
            "Epoch 56/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.3906 - accuracy: 0.8344\n",
            "Epoch 57/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.4068 - accuracy: 0.8339\n",
            "Epoch 58/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.3926 - accuracy: 0.8388\n",
            "Epoch 59/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.3797 - accuracy: 0.8450\n",
            "Epoch 60/75\n",
            "36/36 [==============================] - 1s 25ms/step - loss: 0.3745 - accuracy: 0.8490\n",
            "Epoch 61/75\n",
            "36/36 [==============================] - 1s 28ms/step - loss: 0.3695 - accuracy: 0.8530\n",
            "Epoch 62/75\n",
            "36/36 [==============================] - 1s 22ms/step - loss: 0.3678 - accuracy: 0.8477\n",
            "Epoch 63/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.3638 - accuracy: 0.8552\n",
            "Epoch 64/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.3526 - accuracy: 0.8583\n",
            "Epoch 65/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.3624 - accuracy: 0.8468\n",
            "Epoch 66/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.3461 - accuracy: 0.8574\n",
            "Epoch 67/75\n",
            "36/36 [==============================] - 1s 17ms/step - loss: 0.3359 - accuracy: 0.8578\n",
            "Epoch 68/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.3335 - accuracy: 0.8640\n",
            "Epoch 69/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.3419 - accuracy: 0.8640\n",
            "Epoch 70/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.3215 - accuracy: 0.8742\n",
            "Epoch 71/75\n",
            "36/36 [==============================] - 1s 18ms/step - loss: 0.3347 - accuracy: 0.8632\n",
            "Epoch 72/75\n",
            "36/36 [==============================] - 1s 20ms/step - loss: 0.3183 - accuracy: 0.8773\n",
            "Epoch 73/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.3230 - accuracy: 0.8689\n",
            "Epoch 74/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.3184 - accuracy: 0.8720\n",
            "Epoch 75/75\n",
            "36/36 [==============================] - 1s 19ms/step - loss: 0.2999 - accuracy: 0.8831\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x795a8939d150>"
            ]
          },
          "metadata": {},
          "execution_count": 111
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MR6JtlVa9lh5",
        "outputId": "4165c913-cea9-4f83-88d2-7d379f956ea0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "31/31 [==============================] - 0s 4ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), precision_score(ytest, pred), recall_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YPGXPSru9rFD",
        "outputId": "4d091f4b-4c30-4851-8a07-bb8f5bc716f4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8059855521155831,\n",
              " 0.6126126126126126,\n",
              " 0.3192488262910798,\n",
              " 0.4197530864197531,\n",
              " 0.31686353920575994,\n",
              " 0.34116196032899204)"
            ]
          },
          "metadata": {},
          "execution_count": 113
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AnurtdUR9udh",
        "outputId": "5c2eca8c-29ed-4246-d604-32a0b9a013d8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9431216931216931"
            ]
          },
          "metadata": {},
          "execution_count": 114
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "JeFV86eDUjWY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "g91JSX2gUlcR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "cS_MzYDXUqEF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN()\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "waZx1TlYUsJ-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "C5kzZbvqUubr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "ohNyMOYFUwmm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "QKKL_FGKUy3J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "KHL_AWuEU2Hv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "zvOoKeT_U4s1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "D6z0B8xeU6zB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "R8EmZW0SU8ih"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "gKh8uXmVVAnF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fKBgvCCqVCxp",
        "outputId": "803aa4d3-0dd1-40ce-b4c7-233503d7fa73"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 5s 38ms/step - loss: 0.6925 - accuracy: 0.5218\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 2s 38ms/step - loss: 0.6899 - accuracy: 0.5240\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.6828 - accuracy: 0.5548\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.6788 - accuracy: 0.5760\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.6674 - accuracy: 0.5854\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.6546 - accuracy: 0.6095\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.6377 - accuracy: 0.6332\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.6214 - accuracy: 0.6549\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.5951 - accuracy: 0.6785\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.5734 - accuracy: 0.6963\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.5357 - accuracy: 0.7267\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.5020 - accuracy: 0.7583\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.4582 - accuracy: 0.7799\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.4381 - accuracy: 0.7937\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.4089 - accuracy: 0.8098\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.3841 - accuracy: 0.8342\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.3538 - accuracy: 0.8484\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.3303 - accuracy: 0.8600\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 3s 48ms/step - loss: 0.3090 - accuracy: 0.8756\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 2s 38ms/step - loss: 0.2869 - accuracy: 0.8853\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.2677 - accuracy: 0.8966\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.2442 - accuracy: 0.9091\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.2535 - accuracy: 0.9005\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.2194 - accuracy: 0.9195\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.2079 - accuracy: 0.9219\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.2023 - accuracy: 0.9260\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.2097 - accuracy: 0.9147\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.1869 - accuracy: 0.9287\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.1686 - accuracy: 0.9373\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.1593 - accuracy: 0.9446\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.1410 - accuracy: 0.9530\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.1418 - accuracy: 0.9516\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.1424 - accuracy: 0.9506\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.1187 - accuracy: 0.9605\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.1106 - accuracy: 0.9595\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 3s 50ms/step - loss: 0.1210 - accuracy: 0.9597\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.1370 - accuracy: 0.9571\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.1205 - accuracy: 0.9581\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 0.1047 - accuracy: 0.9646\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.0895 - accuracy: 0.9706\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.0844 - accuracy: 0.9718\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0880 - accuracy: 0.9704\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0909 - accuracy: 0.9720\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0913 - accuracy: 0.9694\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0868 - accuracy: 0.9708\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.0898 - accuracy: 0.9704\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 2s 38ms/step - loss: 0.0790 - accuracy: 0.9769\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 0.0738 - accuracy: 0.9812\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0748 - accuracy: 0.9783\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0677 - accuracy: 0.9831\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0944 - accuracy: 0.9754\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 2s 28ms/step - loss: 0.0666 - accuracy: 0.9829\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0566 - accuracy: 0.9846\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 3s 45ms/step - loss: 0.0484 - accuracy: 0.9877\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0517 - accuracy: 0.9894\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0611 - accuracy: 0.9848\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0437 - accuracy: 0.9879\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0803 - accuracy: 0.9778\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 0.0499 - accuracy: 0.9865\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0332 - accuracy: 0.9920\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0387 - accuracy: 0.9913\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0451 - accuracy: 0.9901\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0416 - accuracy: 0.9889\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 2s 38ms/step - loss: 0.1057 - accuracy: 0.9682\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 3s 47ms/step - loss: 0.0848 - accuracy: 0.9706\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0479 - accuracy: 0.9884\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0357 - accuracy: 0.9930\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.0317 - accuracy: 0.9940\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.0305 - accuracy: 0.9942\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0605 - accuracy: 0.9822\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0523 - accuracy: 0.9855\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0259 - accuracy: 0.9954\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0290 - accuracy: 0.9933\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0396 - accuracy: 0.9933\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0266 - accuracy: 0.9954\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23acc30c70>"
            ]
          },
          "metadata": {},
          "execution_count": 205
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NId4PWgzVEwb",
        "outputId": "790582ef-3ee5-4fc2-d944-3fa6554c0c11"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p8gUDy29VIIl",
        "outputId": "f28b0b5e-1519-4204-db48-f0bb2c7a271e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.973963355834137,\n",
              " 0.9940476190476191,\n",
              " 0.9542857142857143,\n",
              " 0.947944960985214,\n",
              " 0.9487228615187794)"
            ]
          },
          "metadata": {},
          "execution_count": 207
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z9AiKJw3VK0A",
        "outputId": "1100777c-6c3a-435c-9c3c-903b0771d538"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9549718574108818"
            ]
          },
          "metadata": {},
          "execution_count": 208
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "7iRl69fDVN4B"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "5e-j_b1VVR87"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "0n0xs9PKVYFs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek()\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "vN86XrB0Va4J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "vnzkGRHJVdUk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "a0PqvEKPVfJB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "wmp_7VOwViVK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "rV8-23G3VlGa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "9Yhj97VUVnUh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "jABmUau3Vpyw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "ZQXo9Q8UVscC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "LLlmhfquVuoQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S7EOla96VwhK",
        "outputId": "6ba4be6b-10f1-45ad-acb0-5f5a25c23ec7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "65/65 [==============================] - 4s 31ms/step - loss: 0.6931 - accuracy: 0.5094\n",
            "Epoch 2/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.6798 - accuracy: 0.5678\n",
            "Epoch 3/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.6632 - accuracy: 0.6234\n",
            "Epoch 4/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.6518 - accuracy: 0.6309\n",
            "Epoch 5/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.6418 - accuracy: 0.6457\n",
            "Epoch 6/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.6294 - accuracy: 0.6602\n",
            "Epoch 7/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.6089 - accuracy: 0.6730\n",
            "Epoch 8/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.5886 - accuracy: 0.6922\n",
            "Epoch 9/75\n",
            "65/65 [==============================] - 3s 47ms/step - loss: 0.5591 - accuracy: 0.7188\n",
            "Epoch 10/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.5272 - accuracy: 0.7316\n",
            "Epoch 11/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.4951 - accuracy: 0.7609\n",
            "Epoch 12/75\n",
            "65/65 [==============================] - 2s 24ms/step - loss: 0.4791 - accuracy: 0.7643\n",
            "Epoch 13/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.4512 - accuracy: 0.7812\n",
            "Epoch 14/75\n",
            "65/65 [==============================] - 2s 24ms/step - loss: 0.4210 - accuracy: 0.8015\n",
            "Epoch 15/75\n",
            "65/65 [==============================] - 2s 26ms/step - loss: 0.4073 - accuracy: 0.8151\n",
            "Epoch 16/75\n",
            "65/65 [==============================] - 3s 52ms/step - loss: 0.3677 - accuracy: 0.8349\n",
            "Epoch 17/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.3461 - accuracy: 0.8466\n",
            "Epoch 18/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.3184 - accuracy: 0.8623\n",
            "Epoch 19/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.3205 - accuracy: 0.8550\n",
            "Epoch 20/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.2992 - accuracy: 0.8722\n",
            "Epoch 21/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.2839 - accuracy: 0.8758\n",
            "Epoch 22/75\n",
            "65/65 [==============================] - 3s 43ms/step - loss: 0.2705 - accuracy: 0.8860\n",
            "Epoch 23/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.2531 - accuracy: 0.8887\n",
            "Epoch 24/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 0.2348 - accuracy: 0.9020\n",
            "Epoch 25/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.2271 - accuracy: 0.9083\n",
            "Epoch 26/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.2205 - accuracy: 0.9080\n",
            "Epoch 27/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.2085 - accuracy: 0.9177\n",
            "Epoch 28/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.1848 - accuracy: 0.9223\n",
            "Epoch 29/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.1777 - accuracy: 0.9315\n",
            "Epoch 30/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.1655 - accuracy: 0.9359\n",
            "Epoch 31/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.1749 - accuracy: 0.9276\n",
            "Epoch 32/75\n",
            "65/65 [==============================] - 3s 45ms/step - loss: 0.1464 - accuracy: 0.9460\n",
            "Epoch 33/75\n",
            "65/65 [==============================] - 3s 46ms/step - loss: 0.1462 - accuracy: 0.9417\n",
            "Epoch 34/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.1452 - accuracy: 0.9448\n",
            "Epoch 35/75\n",
            "65/65 [==============================] - 2s 27ms/step - loss: 0.1467 - accuracy: 0.9412\n",
            "Epoch 36/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.1437 - accuracy: 0.9465\n",
            "Epoch 37/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.1263 - accuracy: 0.9528\n",
            "Epoch 38/75\n",
            "65/65 [==============================] - 3s 39ms/step - loss: 0.1062 - accuracy: 0.9632\n",
            "Epoch 39/75\n",
            "65/65 [==============================] - 3s 42ms/step - loss: 0.1079 - accuracy: 0.9606\n",
            "Epoch 40/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.1059 - accuracy: 0.9613\n",
            "Epoch 41/75\n",
            "65/65 [==============================] - 2s 31ms/step - loss: 0.1081 - accuracy: 0.9620\n",
            "Epoch 42/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.0800 - accuracy: 0.9760\n",
            "Epoch 43/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0812 - accuracy: 0.9743\n",
            "Epoch 44/75\n",
            "65/65 [==============================] - 2s 33ms/step - loss: 0.0776 - accuracy: 0.9751\n",
            "Epoch 45/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.0748 - accuracy: 0.9777\n",
            "Epoch 46/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0743 - accuracy: 0.9787\n",
            "Epoch 47/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0715 - accuracy: 0.9785\n",
            "Epoch 48/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0728 - accuracy: 0.9794\n",
            "Epoch 49/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.0546 - accuracy: 0.9862\n",
            "Epoch 50/75\n",
            "65/65 [==============================] - 3s 53ms/step - loss: 0.0609 - accuracy: 0.9835\n",
            "Epoch 51/75\n",
            "65/65 [==============================] - 2s 35ms/step - loss: 0.0687 - accuracy: 0.9746\n",
            "Epoch 52/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0532 - accuracy: 0.9848\n",
            "Epoch 53/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0851 - accuracy: 0.9685\n",
            "Epoch 54/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0583 - accuracy: 0.9811\n",
            "Epoch 55/75\n",
            "65/65 [==============================] - 3s 44ms/step - loss: 0.0818 - accuracy: 0.9751\n",
            "Epoch 56/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0533 - accuracy: 0.9845\n",
            "Epoch 57/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.0507 - accuracy: 0.9867\n",
            "Epoch 58/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0461 - accuracy: 0.9857\n",
            "Epoch 59/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.0641 - accuracy: 0.9823\n",
            "Epoch 60/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0371 - accuracy: 0.9896\n",
            "Epoch 61/75\n",
            "65/65 [==============================] - 4s 56ms/step - loss: 0.0343 - accuracy: 0.9918\n",
            "Epoch 62/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0271 - accuracy: 0.9949\n",
            "Epoch 63/75\n",
            "65/65 [==============================] - 2s 37ms/step - loss: 0.0365 - accuracy: 0.9908\n",
            "Epoch 64/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0305 - accuracy: 0.9939\n",
            "Epoch 65/75\n",
            "65/65 [==============================] - 3s 40ms/step - loss: 0.0289 - accuracy: 0.9935\n",
            "Epoch 66/75\n",
            "65/65 [==============================] - 3s 49ms/step - loss: 0.0275 - accuracy: 0.9930\n",
            "Epoch 67/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0313 - accuracy: 0.9932\n",
            "Epoch 68/75\n",
            "65/65 [==============================] - 2s 36ms/step - loss: 0.0479 - accuracy: 0.9838\n",
            "Epoch 69/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0444 - accuracy: 0.9891\n",
            "Epoch 70/75\n",
            "65/65 [==============================] - 2s 32ms/step - loss: 0.0224 - accuracy: 0.9959\n",
            "Epoch 71/75\n",
            "65/65 [==============================] - 2s 34ms/step - loss: 0.0353 - accuracy: 0.9952\n",
            "Epoch 72/75\n",
            "65/65 [==============================] - 3s 41ms/step - loss: 0.0267 - accuracy: 0.9935\n",
            "Epoch 73/75\n",
            "65/65 [==============================] - 2s 29ms/step - loss: 0.0230 - accuracy: 0.9954\n",
            "Epoch 74/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0318 - accuracy: 0.9927\n",
            "Epoch 75/75\n",
            "65/65 [==============================] - 2s 30ms/step - loss: 0.0451 - accuracy: 0.9869\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23ac8bfee0>"
            ]
          },
          "metadata": {},
          "execution_count": 220
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sh1PKtK2Vy1R",
        "outputId": "8867601e-9dc9-432a-a5fd-874d3ae6bce6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "33/33 [==============================] - 0s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JQFI9pNIV2Ua",
        "outputId": "c188adac-c92e-48f2-fc9e-065bf23ee4c7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.936046511627907,\n",
              " 0.9523809523809523,\n",
              " 0.9242144177449169,\n",
              " 0.8719848438715789,\n",
              " 0.8724050551051603)"
            ]
          },
          "metadata": {},
          "execution_count": 222
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dl3lj558V5M9",
        "outputId": "0790ade9-156b-4f96-ad4f-21b4b158bc93"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9191321499013807"
            ]
          },
          "metadata": {},
          "execution_count": 223
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "K-Lzr7mrV7QL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')"
      ],
      "metadata": {
        "id": "EaK6sgRAV_5i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "JfEe7R86WFgc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "DDDUoAk5WHoA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = X.to_numpy()\n",
        "X = X.reshape(X.shape[0], X.shape[1], 1)"
      ],
      "metadata": {
        "id": "x-Iztqk2WJvo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(X):\n",
        "    X_train, X_val = X[train_index], X[val_index]\n",
        "    y_train, y_val = Y[train_index], Y[val_index]"
      ],
      "metadata": {
        "id": "yJC5fH-rWL_4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann = Sequential()"
      ],
      "metadata": {
        "id": "mUM4H-BJWOcz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(256, activation = 'relu', input_shape=(X_train.shape[1], 1)))\n",
        "ann.add(Dense(256, activation = 'relu'))\n",
        "ann.add(Dense(128, activation = 'relu'))"
      ],
      "metadata": {
        "id": "2K83JirYWQoK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "X-BHpo3IWSm-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Flatten())"
      ],
      "metadata": {
        "id": "ms9X9d4mWUgp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.add(Dense(64, activation='relu'))\n",
        "ann.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "y0YMP9gxWW7I"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "JyeyjjSTWY2w"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ann.fit(X_train, y_train, epochs = 75, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q-pWuwSHWbZZ",
        "outputId": "aefeee72-6e93-406c-8fe9-f1f6434574cb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/75\n",
            "17/17 [==============================] - 4s 32ms/step - loss: 0.6935 - accuracy: 0.4797\n",
            "Epoch 2/75\n",
            "17/17 [==============================] - 1s 43ms/step - loss: 0.6904 - accuracy: 0.5378\n",
            "Epoch 3/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.6925 - accuracy: 0.5320\n",
            "Epoch 4/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.6841 - accuracy: 0.6066\n",
            "Epoch 5/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.6805 - accuracy: 0.5833\n",
            "Epoch 6/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.6592 - accuracy: 0.6337\n",
            "Epoch 7/75\n",
            "17/17 [==============================] - 1s 37ms/step - loss: 0.6151 - accuracy: 0.6696\n",
            "Epoch 8/75\n",
            "17/17 [==============================] - 1s 43ms/step - loss: 0.5768 - accuracy: 0.7171\n",
            "Epoch 9/75\n",
            "17/17 [==============================] - 1s 40ms/step - loss: 0.5167 - accuracy: 0.7481\n",
            "Epoch 10/75\n",
            "17/17 [==============================] - 1s 43ms/step - loss: 0.4805 - accuracy: 0.7636\n",
            "Epoch 11/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.4578 - accuracy: 0.7810\n",
            "Epoch 12/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.4450 - accuracy: 0.7897\n",
            "Epoch 13/75\n",
            "17/17 [==============================] - 1s 47ms/step - loss: 0.4450 - accuracy: 0.7907\n",
            "Epoch 14/75\n",
            "17/17 [==============================] - 1s 45ms/step - loss: 0.4427 - accuracy: 0.7878\n",
            "Epoch 15/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.4394 - accuracy: 0.7800\n",
            "Epoch 16/75\n",
            "17/17 [==============================] - 1s 38ms/step - loss: 0.4330 - accuracy: 0.8043\n",
            "Epoch 17/75\n",
            "17/17 [==============================] - 1s 43ms/step - loss: 0.4418 - accuracy: 0.8004\n",
            "Epoch 18/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.4585 - accuracy: 0.7868\n",
            "Epoch 19/75\n",
            "17/17 [==============================] - 1s 35ms/step - loss: 0.4342 - accuracy: 0.7926\n",
            "Epoch 20/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.4191 - accuracy: 0.8101\n",
            "Epoch 21/75\n",
            "17/17 [==============================] - 1s 52ms/step - loss: 0.4275 - accuracy: 0.7936\n",
            "Epoch 22/75\n",
            "17/17 [==============================] - 1s 56ms/step - loss: 0.4163 - accuracy: 0.8081\n",
            "Epoch 23/75\n",
            "17/17 [==============================] - 1s 47ms/step - loss: 0.4156 - accuracy: 0.8101\n",
            "Epoch 24/75\n",
            "17/17 [==============================] - 1s 35ms/step - loss: 0.4203 - accuracy: 0.7955\n",
            "Epoch 25/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.4219 - accuracy: 0.8004\n",
            "Epoch 26/75\n",
            "17/17 [==============================] - 1s 29ms/step - loss: 0.4072 - accuracy: 0.8149\n",
            "Epoch 27/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.4120 - accuracy: 0.8091\n",
            "Epoch 28/75\n",
            "17/17 [==============================] - 1s 38ms/step - loss: 0.4090 - accuracy: 0.8169\n",
            "Epoch 29/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.4193 - accuracy: 0.8081\n",
            "Epoch 30/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.4001 - accuracy: 0.8188\n",
            "Epoch 31/75\n",
            "17/17 [==============================] - 1s 35ms/step - loss: 0.3933 - accuracy: 0.8130\n",
            "Epoch 32/75\n",
            "17/17 [==============================] - 1s 35ms/step - loss: 0.3885 - accuracy: 0.8149\n",
            "Epoch 33/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.3878 - accuracy: 0.8256\n",
            "Epoch 34/75\n",
            "17/17 [==============================] - 1s 33ms/step - loss: 0.3905 - accuracy: 0.8198\n",
            "Epoch 35/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.4045 - accuracy: 0.8140\n",
            "Epoch 36/75\n",
            "17/17 [==============================] - 1s 34ms/step - loss: 0.3763 - accuracy: 0.8207\n",
            "Epoch 37/75\n",
            "17/17 [==============================] - 1s 34ms/step - loss: 0.3766 - accuracy: 0.8333\n",
            "Epoch 38/75\n",
            "17/17 [==============================] - 1s 33ms/step - loss: 0.3809 - accuracy: 0.8275\n",
            "Epoch 39/75\n",
            "17/17 [==============================] - 1s 34ms/step - loss: 0.3658 - accuracy: 0.8382\n",
            "Epoch 40/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.3623 - accuracy: 0.8324\n",
            "Epoch 41/75\n",
            "17/17 [==============================] - 1s 40ms/step - loss: 0.3653 - accuracy: 0.8343\n",
            "Epoch 42/75\n",
            "17/17 [==============================] - 1s 48ms/step - loss: 0.3679 - accuracy: 0.8295\n",
            "Epoch 43/75\n",
            "17/17 [==============================] - 1s 53ms/step - loss: 0.3618 - accuracy: 0.8324\n",
            "Epoch 44/75\n",
            "17/17 [==============================] - 1s 49ms/step - loss: 0.4556 - accuracy: 0.7713\n",
            "Epoch 45/75\n",
            "17/17 [==============================] - 1s 36ms/step - loss: 0.3762 - accuracy: 0.8353\n",
            "Epoch 46/75\n",
            "17/17 [==============================] - 1s 45ms/step - loss: 0.3730 - accuracy: 0.8207\n",
            "Epoch 47/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.3545 - accuracy: 0.8372\n",
            "Epoch 48/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3558 - accuracy: 0.8372\n",
            "Epoch 49/75\n",
            "17/17 [==============================] - 1s 33ms/step - loss: 0.3487 - accuracy: 0.8479\n",
            "Epoch 50/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.3380 - accuracy: 0.8391\n",
            "Epoch 51/75\n",
            "17/17 [==============================] - 1s 36ms/step - loss: 0.3594 - accuracy: 0.8304\n",
            "Epoch 52/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.3368 - accuracy: 0.8401\n",
            "Epoch 53/75\n",
            "17/17 [==============================] - 1s 43ms/step - loss: 0.3290 - accuracy: 0.8469\n",
            "Epoch 54/75\n",
            "17/17 [==============================] - 1s 38ms/step - loss: 0.3287 - accuracy: 0.8556\n",
            "Epoch 55/75\n",
            "17/17 [==============================] - 1s 38ms/step - loss: 0.3144 - accuracy: 0.8624\n",
            "Epoch 56/75\n",
            "17/17 [==============================] - 1s 35ms/step - loss: 0.3128 - accuracy: 0.8527\n",
            "Epoch 57/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3091 - accuracy: 0.8605\n",
            "Epoch 58/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.3660 - accuracy: 0.8314\n",
            "Epoch 59/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3479 - accuracy: 0.8440\n",
            "Epoch 60/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3296 - accuracy: 0.8479\n",
            "Epoch 61/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.3236 - accuracy: 0.8517\n",
            "Epoch 62/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.3132 - accuracy: 0.8585\n",
            "Epoch 63/75\n",
            "17/17 [==============================] - 1s 34ms/step - loss: 0.3302 - accuracy: 0.8459\n",
            "Epoch 64/75\n",
            "17/17 [==============================] - 1s 44ms/step - loss: 0.2938 - accuracy: 0.8624\n",
            "Epoch 65/75\n",
            "17/17 [==============================] - 1s 41ms/step - loss: 0.3007 - accuracy: 0.8595\n",
            "Epoch 66/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.2852 - accuracy: 0.8721\n",
            "Epoch 67/75\n",
            "17/17 [==============================] - 1s 35ms/step - loss: 0.2986 - accuracy: 0.8576\n",
            "Epoch 68/75\n",
            "17/17 [==============================] - 1s 32ms/step - loss: 0.2994 - accuracy: 0.8643\n",
            "Epoch 69/75\n",
            "17/17 [==============================] - 0s 28ms/step - loss: 0.2767 - accuracy: 0.8702\n",
            "Epoch 70/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.2645 - accuracy: 0.8895\n",
            "Epoch 71/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.2774 - accuracy: 0.8847\n",
            "Epoch 72/75\n",
            "17/17 [==============================] - 0s 29ms/step - loss: 0.2943 - accuracy: 0.8614\n",
            "Epoch 73/75\n",
            "17/17 [==============================] - 1s 31ms/step - loss: 0.2585 - accuracy: 0.8876\n",
            "Epoch 74/75\n",
            "17/17 [==============================] - 0s 27ms/step - loss: 0.2662 - accuracy: 0.8924\n",
            "Epoch 75/75\n",
            "17/17 [==============================] - 1s 30ms/step - loss: 0.2611 - accuracy: 0.8847\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7c23a9551750>"
            ]
          },
          "metadata": {},
          "execution_count": 235
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = ann.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mzw-rS-TWdZc",
        "outputId": "430df70b-3dfa-4275-e14a-5ab9fb979dc6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9/9 [==============================] - 0s 4ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), recall_score(y_val, y_pred_classes), precision_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Fep90JZ3Wf-1",
        "outputId": "1dfe70c6-1aa4-4a40-e1cf-8b58e308c690"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8255813953488372,\n",
              " 0.8661971830985915,\n",
              " 0.825503355704698,\n",
              " 0.645626030156889,\n",
              " 0.6466092188625107)"
            ]
          },
          "metadata": {},
          "execution_count": 237
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "35ipreLrWknY",
        "outputId": "9b9155a9-4d78-473b-b8de-e5c0afdffeed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7758620689655172"
            ]
          },
          "metadata": {},
          "execution_count": 238
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **MLP(APAAC)**"
      ],
      "metadata": {
        "id": "u976iNwoZMBT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "mlp = MLPClassifier(hidden_layer_sizes=(8,7), learning_rate_init=0.1, random_state= 50)"
      ],
      "metadata": {
        "id": "xuAq4-CPZQMi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)"
      ],
      "metadata": {
        "id": "nC8SOfR1ZWvi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "Qg9UZ1cCZY9K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "ru2mXqhtZbay"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "CYJ0AwttZhxz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mnN13OyWZjyK",
        "outputId": "c02b5e35-08b1-4bca-ecc4-490b1b2dab86"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.869228385497366,\n",
              " 0.6713178294573643,\n",
              " 0.6734059097978227,\n",
              " 0.5906725347815658,\n",
              " 0.5906736462442057)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Kk-ozJtKZmLq",
        "outputId": "bf58752f-bfdf-4423-ef5f-9b33be14ed4b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9186676994577847"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "2B-xiDHxojzc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "e_HW3Np3olKs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "4t-8NZXXom2L"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mlp.fit(xtrain, ytrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 92
        },
        "id": "DMWQMQH6oqf7",
        "outputId": "eb6dbb82-135f-4225-cf94-f7b8f613ed6f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 83
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = mlp.predict(xtest)"
      ],
      "metadata": {
        "id": "7RiTdEMeowzr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), recall_score(ytest, pred), precision_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kMaOf0hBo38U",
        "outputId": "8e6ea72c-9891-4a10-bf37-b34b7e50788b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8555211558307534,\n",
              " 0.509009009009009,\n",
              " 0.7847222222222222,\n",
              " 0.6174863387978142,\n",
              " 0.5333622273130663,\n",
              " 0.5523561245475603)"
            ]
          },
          "metadata": {},
          "execution_count": 85
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7Owxfchno-wz",
        "outputId": "df7f7ab3-a0ed-4b6f-9779-2024579ccd7a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9585006693440429"
            ]
          },
          "metadata": {},
          "execution_count": 86
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "MrvoNWElZqHP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "Htg83JGMZsA8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "Tqp8tvFZZ6Wy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "TsVBZ9yAZ9Hs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w0lGDHhfZ-wC",
        "outputId": "71064d9b-320b-4d89-f55e-6b6464891ba9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8652259332023575,\n",
              " 0.9641148325358851,\n",
              " 0.8022561380225613,\n",
              " 0.7311720999301241,\n",
              " 0.7459848204485944)"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mddfjYDVaBSi",
        "outputId": "7223bd58-ff0a-472f-9d3f-bcc40a58cd07"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7691711851278079"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "0PUIqu6haEE0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "cRVFn_TPaGfs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "-GRYQ7zxaK3y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "evn6TXiUaNja"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kZpgrdZGaPrC",
        "outputId": "f3b19140-0d32-4a0a-c3a5-da68049a60c7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8946553059643687,\n",
              " 0.9539116963594113,\n",
              " 0.8528393351800554,\n",
              " 0.7893106119287374,\n",
              " 0.7949127380218401)"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fNofsRlxaSPs",
        "outputId": "5cde64ce-9bc9-4331-c858-28fa3f538f20"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8353989155693261"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "TtGuWug3aVTS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/APAAC-TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "p-U0PlSOaXcE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "TJV94js4abtk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "3yp06CS9adgi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RvPgb_Mbagdd",
        "outputId": "0f283dc7-3fef-4818-a65c-c92295c52482"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.7953488372093023,\n",
              " 0.8217054263565892,\n",
              " 0.780559646539028,\n",
              " 0.5906976744186047,\n",
              " 0.5915200683533167)"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RXOj5Nn0ai00",
        "outputId": "0382be11-47c7-4bbd-ef13-7f1790a0ac98"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7689922480620155"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **MLP(Geary)**"
      ],
      "metadata": {
        "id": "fZ_RE4JEamAc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Imbalanced**"
      ],
      "metadata": {
        "id": "SR256tmeavCb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "R2Ba9Dqvapuz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "Sx8IIv--a5kc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UdqQiKv5a7Xq",
        "outputId": "7b29dceb-9b3b-42c2-a14a-b525f63d7dbe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, msg_start, len(result))\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8001239541369694, 0.0, 0.0, 0.0, 0.0)"
            ]
          },
          "metadata": {},
          "execution_count": 123
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_FWFAyZna9hd",
        "outputId": "8a7afbfb-2865-40ff-939b-9fff29b49afd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 124
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "mwflE07E-ekH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "8_2YCMJm-fuJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)"
      ],
      "metadata": {
        "id": "-iI3SGxD-oHZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mlp.fit(xtrain, ytrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 92
        },
        "id": "-8-l52RY-q0R",
        "outputId": "715af11b-c35c-47b9-9093-ed976307da09"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)"
            ],
            "text/html": [
              "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier(hidden_layer_sizes=(8, 7), learning_rate_init=0.1,\n",
              "              random_state=50)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 127
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = mlp.predict(xtest)"
      ],
      "metadata": {
        "id": "qacq49zN-3xS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), recall_score(ytest, pred), precision_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8FrZgAnB-3F7",
        "outputId": "4c930eeb-2126-4e4e-efa3-34ed55a71c1a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.7791537667698658,\n",
              " 0.0,\n",
              " 0.0,\n",
              " 0.0,\n",
              " -0.0020585677007827208,\n",
              " -0.01706047918878187)"
            ]
          },
          "metadata": {},
          "execution_count": 129
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rm1FKCHR-6hL",
        "outputId": "32b25726-df42-4ea5-c714-58680d8ed5d0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9986772486772487"
            ]
          },
          "metadata": {},
          "execution_count": 130
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADASYN**"
      ],
      "metadata": {
        "id": "v3VxiuadbC6r"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "zFDwnkrkbFEs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import ADASYN\n",
        "ada = ADASYN(random_state=42)\n",
        "X, Y = ada.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "APhdy-6tbJei"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "hW9VRLg7bMi0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C60QyAdCbOlr",
        "outputId": "5aeceae3-8aca-40d8-d625-ec1429726c77"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.5466640956421134,\n",
              " 0.5898617511520737,\n",
              " 0.54487406881873,\n",
              " 0.09299283472220499,\n",
              " 0.09331439372247223)"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "v1cLP9_8bRH0",
        "outputId": "8cfc574f-6fe9-4d6d-bfdf-13ed71d7e78b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.5030983733539891"
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SMOTETomek**"
      ],
      "metadata": {
        "id": "iEK-Dk_RbVL9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "xjeHsCncbYTy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.combine import SMOTETomek\n",
        "smt = SMOTETomek(random_state=42)\n",
        "X, Y = smt.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "R-UQiG5zbawu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "FN7H55rTbgwK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A74A1f4Bbjny",
        "outputId": "82fbdc6a-511b-463e-c61b-395a8d2b4ba1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.617157242447715,\n",
              " 0.4721146398140976,\n",
              " 0.6650300054555374,\n",
              " 0.23431448489542994,\n",
              " 0.24484251158488696)"
            ]
          },
          "metadata": {},
          "execution_count": 38
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z4-8PiohbmaD",
        "outputId": "c5286b97-84a9-40dd-9808-01bfcc2bc997"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7621998450813323"
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**NearMiss**"
      ],
      "metadata": {
        "id": "Y-0F3iaYbp-E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = pd.read_csv('/content/Geary_TR.csv')\n",
        "columns = df1.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df1[columns]\n",
        "Y = df1[target]"
      ],
      "metadata": {
        "id": "2xHbtgHmbr2C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.under_sampling import NearMiss\n",
        "nm = NearMiss()\n",
        "X, Y = nm.fit_resample(X, Y)"
      ],
      "metadata": {
        "id": "Tg8lTRD0bwEz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cross_val_predict(mlp, X, Y, cv=cv, n_jobs=-1)"
      ],
      "metadata": {
        "id": "mRLCZCRFbyRM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(Y, pred), recall_score(Y, pred), precision_score(Y, pred), cohen_kappa_score(Y, pred), matthews_corrcoef(Y, pred)"
      ],
      "metadata": {
        "id": "aH4cvkpwb0O8",
        "outputId": "3b601fb1-578d-4514-e21c-dc0991aab6c4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.5875968992248062,\n",
              " 0.4496124031007752,\n",
              " 0.6209850107066381,\n",
              " 0.17519379844961236,\n",
              " 0.18227205315272715)"
            ]
          },
          "metadata": {},
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(Y, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity"
      ],
      "metadata": {
        "id": "1Zg8UaXsb2hT",
        "outputId": "459929e7-d5fe-4259-bfe8-9561a4b8ccca",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7255813953488373"
            ]
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    }
  ]
}
