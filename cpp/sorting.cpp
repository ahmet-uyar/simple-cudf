//
// Created by auyar on 12.10.2021.
//

/**
 * Sorting a cudf table vs merging multiple sorted tables
 * both with the same number of rows
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/sorting.hpp>
#include <cudf/merge.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <cuda.h>

#include "construct.h"

using namespace std;
using namespace std::chrono;

int64_t calculateRows(const std::string& dataSize, int cols) {
  char last_char = dataSize[dataSize.size() - 1];
  char prev_char = dataSize[dataSize.size() - 2];
  int64_t size_num = stoi(dataSize.substr(0, dataSize.size() - 2));
  if (prev_char == 'M' && last_char == 'B') {
    size_num *= 1000000;
  } else if (prev_char == 'G' && last_char == 'B') {
    size_num *= 1000000000;
  } else {
    throw "data size has to end with either MB or GB!";
  }

  return size_num / (cols * 8);
}

void writeToFile(cudf::table_view &tv, std::string file_name) {
  cudf::io::sink_info sinkInfo(file_name);
  cudf::io::csv_writer_options writerOptions = cudf::io::csv_writer_options::builder(sinkInfo, tv);
  cudf::io::write_csv(writerOptions);
  cout  << "written the table to the file: " << file_name << endl;
}

/**
 * convert a vector of elements to a string with comma + space in between
 * @tparam T
 * @param vec
 * @return
 */
template<typename T>
std::string vectorToString(const std::vector<T> &vec) {
  if (vec.empty()) {
    return std::string();
  }

  std::ostringstream oss;
  // Convert all but the last element to avoid a trailing ","
  std::copy(vec.begin(), vec.end()-1,
            std::ostream_iterator<T>(oss, ", "));

  // Now add the last element with no delimiter
  oss << vec.back();
  return oss.str();
}

std::vector<cudf::table_view> tableSlices(int cols, int64_t rows, int num_slices, std::unique_ptr<cudf::table> &tbl) {
  tbl = constructRandomDataTable(cols, rows);
  auto input_tv = tbl->view();
  cout << "initial dataframe: cols: " << input_tv.num_columns() << ", rows: " << input_tv.num_rows() << endl;
//  writeToFile(tv, "initial_table.csv");

  std::vector<cudf::size_type> slice_rows;

  int64_t rows_per_table = rows / num_slices;
  int32_t row_start = 0;
  for (int i = 0; i < num_slices - 1; ++i) {
    slice_rows.push_back(row_start);
    slice_rows.push_back(row_start + rows_per_table);
    row_start += rows_per_table;
  }
  // add the last slice indices
  slice_rows.push_back(row_start);
  slice_rows.push_back(input_tv.num_rows());

//  cout << "slice indices: " << vectorToString(slice_rows) << endl;

  auto init_tv_slices = cudf::slice(input_tv, slice_rows);
  cout << "number of sliced tables: " << init_tv_slices.size() << endl;
  return init_tv_slices;
}

int testSorting(int cols, int64_t rows, int num_tables) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::unique_ptr<cudf::table> tbl;
  auto init_tv_slices = tableSlices(cols, rows, num_tables, tbl);

  vector<unique_ptr<cudf::table>> separate_tables;
  std::vector<cudf::table_view> separate_tviews;
  for (auto tv_slice: init_tv_slices) {
    unique_ptr<cudf::table> tbl = make_unique<cudf::table>(tv_slice);
    separate_tviews.push_back(tbl->view());
    separate_tables.push_back(move(tbl));
  }

  // concat and sort
  cudaEventRecord(start);
  auto single_tbl = cudf::concatenate(separate_tviews);
  std::unique_ptr<cudf::table> result_table = cudf::sort(single_tbl->view());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float fdelay = 0;
  cudaEventElapsedTime(&fdelay, start, stop);
  int delay = (int)fdelay;

  auto result_tv = result_table->view();

  cout << "duration: "<<  delay << endl;
//  writeToFile(result_tv, "sorted_table.csv");
  cout << "rows in sorted df: "<< result_tv.num_rows()  << endl;
  return delay;
}

int testMerging(const int cols, const int64_t rows, int num_of_sorted_tables) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::unique_ptr<cudf::table> tbl;
  auto init_tv_slices = tableSlices(cols, rows, num_of_sorted_tables, tbl);
  vector<unique_ptr<cudf::table>> sorted_tables;
  vector<cudf::table_view> sorted_tvs;
  for (auto tv: init_tv_slices) {
    auto sorted_tbl = cudf::sort(tv);
    sorted_tvs.push_back(sorted_tbl->view());
    sorted_tables.push_back(move(sorted_tbl));
  }

  std::vector<cudf::order> column_orders;
  std::vector<cudf::size_type> key_cols;
  for (int i = 0; i < cols; ++i) {
    key_cols.push_back(i);
    column_orders.push_back(cudf::order::ASCENDING);
  }

  cudaEventRecord(start);
  std::unique_ptr<cudf::table> result_table = cudf::merge(sorted_tvs, key_cols, column_orders);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float fdelay = 0;
  cudaEventElapsedTime(&fdelay, start, stop);
  int delay = (int)fdelay;

  auto result_tv = result_table->view();
  vector<cudf::null_order> null_orders(result_tv.num_columns(), cudf::null_order::AFTER);
  if(cudf::is_sorted(result_tv, column_orders, null_orders))
    cout << "the table is sorted" << endl;
  else
    cout << "the table is not sorted." << endl;

  cout << "duration: "<< delay << endl;
//  cout << "sorted dataframe................................. " << endl;
  cout << "rows in merged df: "<< result_tv.num_rows() << endl;
  return delay;
}


const bool RESULT_TO_FILE = true;
string OUT_FILE = "single_run.csv";

int main(int argc, char *argv[]) {

  if (argc < 5) {
    cout << "required three params (sort/merge dataSize num_columns num_tables): sorting 1GB 2 4, \n"
               << "dataSize in MB or GB: 100MB, 2GB, etc." << endl;
    return 1;
  }

  std::string op_type = argv[1];
  if(op_type != "sort" && op_type != "merge") {
    cout << "first parameter can be either 'sort' or 'merge'" << endl;
    return 1;
  }

  std::string dataSize = argv[2];
  int cols = stoi(argv[3]);
  int num_of_tables = stoi(argv[4]);

  int number_of_GPUs;
  cudaGetDeviceCount(&number_of_GPUs);

  // set the gpu
//    cudaSetDevice();
  int deviceInUse = -1;
  cudaGetDevice(&deviceInUse);
  cout << "device in use: "<< deviceInUse << ", number of GPUs: " << number_of_GPUs << endl;

  // calculate the number of rows
  int64_t rows = calculateRows(dataSize, cols);
  cout << "number of columns: "<< cols << ", total number of rows: " << rows << endl;

  int delay = -1;
  if(op_type == "sort")
    delay = testSorting(cols, rows, num_of_tables);
  else if(op_type == "merge")
    delay = testMerging(cols, rows, num_of_tables);
//  else if(op_type == "quantiles")
//    delay = testQuantiles(cols, rows);

  if (RESULT_TO_FILE) {
    std::ofstream srf;
    srf.open(OUT_FILE);
    srf << delay << endl;
    srf.close();
  }

  return 0;
}
