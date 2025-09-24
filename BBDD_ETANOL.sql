-- Crear la base de datos (si no existe)
CREATE DATABASE IF NOT EXISTS ETANOL;
USE ETANOL;

-- Tabla Samples: Almacena información sobre las muestras
CREATE TABLE Samples (
    id INT AUTO_INCREMENT PRIMARY KEY,    -- Clave primaria autoincremental
    name VARCHAR(255) UNIQUE NOT NULL,    -- Nombre del archivo CSV
    round INT NOT NULL,                   -- Número de ronda
    concentration FLOAT NOT NULL,          -- Concentración
    sample INT NOT NULL,                   -- Número de muestra
    date DATETIME NOT NULL,                -- Fecha y hora de la muestra
    UNIQUE(round, concentration, sample)   -- Garantiza unicidad combinada
);

-- Tabla Measurements: Almacena las mediciones de cada muestra
CREATE TABLE Measurements (
    id INT AUTO_INCREMENT PRIMARY KEY,     -- Clave primaria autoincremental
    sample_id INT NOT NULL,                -- Clave foránea a Samples
    frequency FLOAT NOT NULL,               -- Frecuencia en Hz
    real_part FLOAT NOT NULL,               -- Parte real
    imaginary_part FLOAT NOT NULL,          -- Parte imaginaria
    FOREIGN KEY (sample_id) REFERENCES Samples(id) ON DELETE CASCADE
);

-- Índices para mejorar el rendimiento en consultas frecuentes
CREATE INDEX idx_samples_round ON Samples(round);
CREATE INDEX idx_samples_concentration ON Samples(concentration);
CREATE INDEX idx_samples_date ON Samples(date);
CREATE INDEX idx_measurements_frequency ON Measurements(frequency);
