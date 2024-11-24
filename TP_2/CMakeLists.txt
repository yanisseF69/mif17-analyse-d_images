cmake_minimum_required(VERSION 3.12)
project(TP2)

# Trouver le package OpenCV
find_package(OpenCV REQUIRED)

# Ajouter l'exécutable de votre projet avec le nom TP2
add_executable(TP2 TP2.cpp)

# Liens avec les bibliothèques OpenCV
target_link_libraries(TP2 ${OpenCV_LIBS})
