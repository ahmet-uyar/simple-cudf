//
// Created by auyar on 17.08.2021.
//
#include <iostream>
#include <cudf/io/csv.hpp>
#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

void printMinMax(const cudf::column_view &cv, int i) {
    std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>> minmax = cudf::minmax(cv);
    if (cudf::is_numeric(cv.type())) {
        if (cv.type().id() == cudf::type_id::INT64) {
            std::unique_ptr<cudf::numeric_scalar<int64_t>> min(static_cast<cudf::numeric_scalar<int64_t> *>(minmax.first.release()));
            std::unique_ptr<cudf::numeric_scalar<int64_t>> max(static_cast<cudf::numeric_scalar<int64_t> *>(minmax.second.release()));
            std::cout << "column[" << i << "] is numeric. min: " << min->value() << ", max: " << max->value() << std::endl;
        }
    } else if(cv.type().id() == cudf::type_id::STRING) {
        std::unique_ptr<cudf::string_scalar> min(static_cast<cudf::string_scalar *>(minmax.first.release()));
        std::unique_ptr<cudf::string_scalar> max(static_cast<cudf::string_scalar *>(minmax.second.release()));
        std::cout << "column[" << i << "] is string. min: " << min->to_string() << ", max: " << max->to_string() << std::endl;
    } else {
        std::cout << "unrecognized column type:" << static_cast<std::underlying_type<cudf::type_id>::type>(cv.type().id()) << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "CUDF Table Read Example" << std::endl;

    std::string input_csv_file = "data/cities.csv";
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    cudf::table_view tv = ctable.tbl->view();

    std::cout << "csv file: " << input_csv_file << ", number of columns: " << tv.num_columns()
        << ", number of rows: " << tv.num_rows() << std::endl;

    // calculate min and max
    for (int i = 0; i < tv.num_columns(); ++i) {
        printMinMax(tv.column(i), i);
    }

    return 0;
}
