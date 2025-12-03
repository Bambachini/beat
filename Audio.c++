```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// Función para preprocesar los datos
void preprocessData(std::vector<double>& signal, int sampleRate) {
    // Saltar los primeros 2 segundos de la señal
    int skipSamples = 2 * sampleRate;
    signal.erase(signal.begin(), signal.begin() + skipSamples);

    // Eliminar el ruido de alta frecuencia (>=40hz)
    // Eliminar el vagabundeo de la línea base (<=0.5hz)
    // Corregir la inversión
    // Filtro de ruido de alta frecuencia (>=40Hz) y compensación de deriva de línea base (<=0.5Hz)

#include <iostream>
#include <vector>
#include <cmath>

// Filtro pasa bajos para la deriva de línea base
std::vector<double> baselineWanderFilter(const std::vector<double>& signal, double cutoffFreq, double samplingFreq) {
std::vector<double> filteredSignal(signal.size());
double RC = 1.0 / (2 * M_PI * cutoffFreq);
double dt = 1.0 / samplingFreq;
double alpha = dt / (RC + dt);

filteredSignal[0] = signal[0];
for (size_t i = 1; i < signal.size(); ++i) {
    filteredSignal[i] = filteredSignal[i - 1] + alpha * (signal[i] - filteredSignal[i - 1]);
}
return filteredSignal;

}

// Filtro pasa altos para el ruido de alta frecuencia
std::vector<double> highFrequencyNoiseFilter(const std::vector<double>& signal, double cutoffFreq, double samplingFreq) {
std::vector<double> filteredSignal(signal.size());
double RC = 1.0 / (2 * M_PI * cutoffFreq);
double dt = 1.0 / samplingFreq;
double alpha = RC / (RC + dt);

filteredSignal[0] = signal[0];
for (size_t i = 1; i < signal.size(); ++i) {
    filteredSignal[i] = alpha * (filteredSignal[i - 1] + signal[i] - signal[i - 1]);
}
return filteredSignal;

}

int main() {
std::vector<double> signal = { /* señal de entrada */ };
double samplingFreq = 1000.0; // Frecuencia de muestreo en Hz

// Filtrar la deriva de la línea base (<= 0.5Hz)
auto filteredBaseline = baselineWanderFilter(signal, 0.5, samplingFreq);

// Filtrar el ruido de alta frecuencia (>= 40Hz)
auto filteredSignal = highFrequencyNoiseFilter(filteredBaseline, 40, samplingFreq);

// Mostrar la señal filtrada
for (const auto& val : filteredSignal) {
    std::cout << val << std::endl;
}

return 0;

}
    
    // Reducir la tasa de muestreo de 300hz a 100hz
    int downsampledRate = sampleRate / 3;
    std::vector<double> downsampledSignal;
    for (int i = 0; i < signal.size(); i += 3) {
        downsampledSignal.push_back(signal[i]);
    }
    signal = downsampledSignal;
    sampleRate = downsampledRate;
}

// Función para dividir el conjunto de datos en conjuntos de entrenamiento, validación y prueba
void splitDataset(std::vector<std::vector<double>>& dataset, int trainRatio, int validRatio, int testRatio) {
    int totalSamples = dataset.size();
    int trainSamples = totalSamples * trainRatio / 100;
    int validSamples = totalSamples * validRatio / 100;
    int testSamples = totalSamples - trainSamples - validSamples;

    std::vector<std::vector<double>> trainSet, validSet, testSet;
    trainSet = dataset.begin(), trainSet += trainSamples;
    validSet = trainSet, validSet += validSamples;
    testSet = validSet, testSet += testSamples;

    dataset = trainSet;
}

// Función para realizar el etiquetado de los datos
void labelEncoding(std::vector<std::vector<double>>& dataset, const std::vector<int>& labels) {
    if (dataset.size() != labels.size()) {
        throw std::invalid_argument(“El tamaño del conjunto de datos y el conjunto de etiquetas deben coincidir.”);
    }
    for (int i = 0; i < dataset.size(); i++) {
        dataset[i].push_back(labels[i]);
    }
}

// Función para realizar la ampliación de datos
void dataAugmentation(std::vector<std::vector<double>>& dataset) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniformrealdistribution<> dist(0.9, 1.1);

    for (auto& sample : dataset) {
        // Escala de tiempo aleatoria
        double scale = dist(gen);
        for (int i = 0; i < sample.size(); i++) {
            sample[i] *= scale;
        }

        // Ruido aleatorio
        double noise = dist(gen) - 1;
        for (int i = 0; i < sample.size(); i++) {
            sample[i] += noise;
        }

        // Inversión aleatoria (inversión vertical)
        if (std::bernoulli_distribution(0.5)(gen)) {
            for (int i = 0; i < sample.size(); i++) {
                sample[i] = -sample[i];
            }
        }

        // Enmascaramiento aleatorio
        int maskSize = static_cast<int>(sample.size() * 0.1);
        for (int i = 0; i < maskSize; i++) {
            int index = std::uniformintdistribution<>(0, sample.size() - 1)(gen);
            sample[index] = 0;
        }
    }
}

// Función para manejar los valores atípicos
void outlierHandling(std::vector<std::vector<double>>& dataset) {
    for (auto& sample : dataset) {
        double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size();
        double stdDev = std::sqrt(std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size() - mean * mean);
        for (int i = 0; i < sample.size(); i++) {
            if (sample[i] > mean + 3 * stdDev || sample[i] < mean - 3 * stdDev) {
                sample[i] = mean;
            }
        }
    }
}

// Función para convertir los datos a tensores
void convertToTensor(std::vector<std::vector<double>>& dataset) {
    for (auto& sample : dataset) {
        for (int i = 0; i < sample.size(); i++) {
            sample[i] = static_cast<double>(sample[i]);
        }
    }
}

// Función para recortar los datos
void cropping(std::vector<std::vector<double>>& dataset, int targetTime, int targetLength, bool isTrain) {
    for (auto& sample : dataset) {
        if (sample.size() > targetLength) {
            if (isTrain) {
                // Recorte aleatorio
                int startIndex = std::uniformintdistribution<>(0, sample.size() - targetLength)(std::mt19937(std::random_device{}()));
                sample.erase(sample.begin() + startIndex, sample.begin() + startIndex + targetLength);
            } else {
                // Recorte de cabeza
                sample.erase(sample.begin() + targetLength, sample.end());
            }
        } else if (sample.size() < targetLength) {
            // Relleno con ceros
            sample.insert(sample.end(), targetLength - sample.size(), 0.0);
        }
    }
}

// Función para normalizar los datos de voltaje
void voltageNormalization(std::vector<std::vector<double>>& dataset, std::string normalizationType) {
    for (auto& sample : dataset) {
        if (normalizationType == “min-max”) {
            double minVal = *std::min_element(sample.begin(), sample.end());
            double maxVal = *std::max_element(sample.begin(), sample.end());
            for (int i = 0; i < sample.size(); i++) {
                sample[i] = (sample[i] - minVal) / (maxVal - minVal);
            }
        } else if (normalizationType == “mean”) {
            double meanVal = std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size();
            for (int i = 0; i < sample.size(); i++) {
                sample[i] -= meanVal;
            }
        } else if (normalizationType == “z-score”) {
            double meanVal = std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size();
            double stdDev = std::sqrt(std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size() - meanVal * meanVal);
            for (int i = 0; i < sample.size(); i++) {
                sample[i] = (sample[i] - meanVal) / stdDev;
            }
        }
    }
}

// Función para desplegar los datos
void unsqueeze(std::vector<std::vector<double>>& dataset) {
    for (auto& sample : dataset) {
        sample.push_back(0.0); // Agregar una dimensión adicional
    }
}

// Función para manejar el conjunto de datos desequilibrado
void handleImbalancedDataset(std::vector<std::vector<double>>& dataset, const std::vector<int>& labels) {
    std::vector<int> classCounts;
    for (int i = 0; i < labels.size(); i++) {
        while (classCounts.size() <= labels[i]) {
            classCounts.push_back(0);
        }
        classCounts[labels[i]]++;
    }

    std::vector<double> weights;
    for (int i = 0; i < classCounts.size(); i++) {
        weights.push_back(1.0 / classCounts[i]);
    }

    std::vector<int> indices;
    for (int i = 0; i < labels.size(); i++) {
        indices.push_back(i);
    }

    std::randomshuffle(indices.begin(), indices.end(), std::mt19937(std::randomdevice{}()));

    std::vector<std::vector<double>> weightedDataset;
    for (int i = 0; i < indices.size(); i++) {
        weightedDataset.push_back(dataset[indices[i]]);
    }

    dataset = weightedDataset;
}

// Función para particionar el conjunto de datos federado
void partitionFederatedDataset(std::vector<std::vector<double>>& dataset, int numClients, std::string partitioningType) {
    int totalSamples = dataset.size();
    int samplesPerClient = totalSamples / numClients;

    std::vector<std::vector<double>> partitionedDataset;
    for (int i = 0; i < numClients; i++) {
        std::vector<std::vector<double>> clientSet;
        clientSet = dataset.begin() + i * samplesPerClient, clientSet += samplesPerClient;
        partitionedDataset.push_back(clientSet);
    }

    if (partitioningType == “non-iid”) {
        std::vector<double> dirichletParams(numClients, 0.5);
        std::dirichlet_distribution<double> dirichlet(dirichletParams);
        std::vector<int> clientSizes;
        dirichlet(std::mt19937(std::random_device{}()), clientSizes.begin(), clientSizes.end());

        partitionedDataset.clear();
        for (int i = 0; i < numClients; i++) {
            std::vector<std::vector<double>> clientSet;
            clientSet = dataset.begin(), clientSet += clientSizes[i];
            partitionedDataset.push_back(clientSet);
            dataset.erase(dataset.begin(), dataset.begin() + clientSizes[i]);
        }
    }

    dataset = partitionedDataset;
}

int main() {
    std::vector<std::vector<double>> dataset; // Datos de entrada
    std::vector<int> labels; // Etiquetas
    int sampleRate = 300; // Tasa de muestreo original

    // Preprocesar los datos
    preprocessData(dataset, sampleRate);

    // Dividir el conjunto de datos
    splitDataset(dataset, 70, 15, 15);

    // Etiquetar los datos
    labelEncoding(dataset, labels);

    // Ampliar los datos
    dataAugmentation(dataset);

    // Manejar los valores atípicos
    outlierHandling(dataset);

    // Convertir los datos a tensores
    convertToTensor(dataset);

    // Recortar los datos
    cropping(dataset, 58, 5800, true); // Conjunto de entrenamiento
    cropping(dataset, 58, 5800, false); // Conjunto de validación
    cropping(dataset, 58, 5800, false); // Conjunto de prueba

    // Normalizar los datos de voltaje
    voltageNormalization(dataset, “min-max”);

    // Desplegar los datos
    unsqueeze(dataset);

    // Manejar el conjunto de datos desequilibrado
    handleImbalancedDataset(dataset, labels);

    // Particionar el conjunto de datos federado
    partitionFederatedDataset(dataset, 5, “non-iid”);

    return 0;
}
```