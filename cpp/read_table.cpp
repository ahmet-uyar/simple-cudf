//
// Created by auyar on 17.08.2021.
//
#include <iostream>
#include <cudf/io/csv.hpp>
#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

int main(int argc, char** argv) {
    std::cout << "CUDF Table Read Example" << std::endl;

    std::string input_csv_file = "data/cities.csv";
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    cudf::table_view tv = ctable.tbl->view();

    std::cout << "csv file: " << input_csv_file << ", number of columns: " << tv.num_columns()
        << ", number of rows: " << tv.num_rows() << std::endl;

    return 0;
}
