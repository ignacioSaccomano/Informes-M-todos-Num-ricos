#include <iostream>
#include <fstream>
#include<Eigen/Dense>

using namespace Eigen;


std::tuple <double,VectorXd, int> metodoPotencia(MatrixXd mat) {
        
        VectorXd vec(VectorXd::Random(mat.cols()));

        unsigned int iteraciones = 0;

        double avalor=0;
        for(int i =0;i<500000; i++){
            iteraciones ++;
            VectorXd vectorViejo = vec/vec.norm();
            vec = mat * vec;
            vec = vec/ vec.norm();
            avalor = (vec.transpose() * mat * vec );
            avalor = avalor/(vec.transpose() * vec);
            VectorXd dif = vectorViejo - vec;
            if(dif.lpNorm<Infinity>() < 1e-7) break; // Tomamos la norma infinito de la diferencia de los vectores, si es < a epsilon paramos.
        }
        std::tuple <double,VectorXd, unsigned int> res = std::make_tuple(avalor, vec, iteraciones);
        return res;
}

int main (int argc, char**argv){
    // Del archivo de entrada solo leemos la matriz y la cantidad de autovalores/autovectores que queremos hallar
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " input_file autovalores_file autovectores_file iteraciones_file" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* autovalores_file = argv[2];
    const char* autovectores_file = argv[3];
    const char* iteraciones_file = argv[4];

    std::ifstream fin(input_file);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open input file " << input_file << std::endl;
        return 1;
    }

    // Read matrix and vector from file
    int nrows, ncols;
    fin >> nrows >> ncols;

    MatrixXd matriz(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> matriz(i, j);
        }
    }

    unsigned int niter;
    fin >> niter;

    fin.close();

    // Deflacion de Hotelling
    MatrixXd autovectores(ncols, niter);
    VectorXd autovalores(niter);
    VectorXd iteraciones(niter);    // Para el 2
    for(unsigned int i =0;i<niter; i++){ // Calculamos niter autovalores y autovectores asociados y los devolvemos en forma matricial.
        std::tuple <double,VectorXd, int> resultadoMetPot = metodoPotencia(matriz);
        autovalores(i) = std::get<0>(resultadoMetPot);
        autovectores.col(i) = std::get<1>(resultadoMetPot);
        iteraciones(i) = std::get<2>(resultadoMetPot);
        matriz = matriz-autovalores(i)*(autovectores.col(i)*(autovectores.col(i).transpose()));
    }
    // Write result to output file
    std::ofstream fout_avalores(autovalores_file);
    if (!fout_avalores.is_open()) {
        std::cerr << "Error: could not open output file " << autovalores_file << std::endl;
        return 1;
    }

    fout_avalores << autovalores << std::endl;

    fout_avalores.close();

    // Write result to output file
    std::ofstream fout_avectores(autovectores_file);
    if (!fout_avectores.is_open()) {
        std::cerr << "Error: could not open output file " << autovectores_file << std::endl;
        return 1;
    }

    fout_avectores << autovectores << std::endl;

    fout_avectores.close();

    // Write result to output file
    std::ofstream fout_iter(iteraciones_file);
    if (!fout_iter.is_open()) {
        std::cerr << "Error: could not open output file " << iteraciones_file << std::endl;
        return 1;
    }
    fout_iter <<  iteraciones << std::endl;

    fout_iter.close();

    return 0;
}