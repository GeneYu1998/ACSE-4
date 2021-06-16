CXX = g++
CXXFLAGS = -Wall -std=c++11
LDFLAGS =
SOURCE_DIR = src
INCLUDE_DIR = includes
TEST_DIR = tests
PIP = pip install
BUILD_DIR = build
BIN_DIR = bin
ALL_BUILD_DIR = $(BUILD_DIR) $(BIN_DIR)

TEST_BUILD_DIR = $(TEST_DIR)/build
TEST_BIN_DIR = $(TEST_DIR)/bin
ALL_TEST_BUILD_DIR = $(TEST_BUILD_DIR) $(TEST_BIN_DIR)

all: Genetic_Algorithm

Genetic_Algorithm: $(BIN_DIR)/Genetic_Algorithm


$(BIN_DIR)/Genetic_Algorithm: $(BUILD_DIR)/Genetic_Algorithm.o $(BUILD_DIR)/CCircuit.o $(BUILD_DIR)/main.o $(BUILD_DIR)/CUnit.o
	$(CXX) -o $@ $^

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(INCLUDE_DIR)/*.h | directories
	$(CXX) $(CPPFLAGS) -o $@ -c $< $(CXXFLAGS) -I$(INCLUDE_DIR)


clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/*

.PHONY: Genetic_Algorithm all clean

TESTS = test1 test2

runtests: ${TESTS}
	@python3 $(TEST_DIR)/run_tests.py

tests: ${TESTS}

test1: $(TEST_BIN_DIR)/test1

test2: $(TEST_BIN_DIR)/test2

$(TEST_BIN_DIR)/test1: $(TEST_BUILD_DIR)/test1.o $(BUILD_DIR)/CCircuit.o  $(BUILD_DIR)/CUnit.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS)

$(TEST_BIN_DIR)/test2: $(TEST_BUILD_DIR)/test2.o $(BUILD_DIR)/Genetic_Algorithm.o $(BUILD_DIR)/CCircuit.o  $(BUILD_DIR)/CUnit.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS)

$(TEST_BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp $(INCLUDE_DIR)/*.h | test_directories
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(CPPFLAGS) -I$(INCLUDE_DIR)

cleantest:
	rm -f $(TEST_BUILD_DIR)/* $(TEST_BIN_DIR)/*

.PHONY: tests ${TESTS} cleantests runtests


directories:
	@mkdir -p $(ALL_BUILD_DIR)

test_directories:
	@mkdir -p $(ALL_TEST_BUILD_DIR)

.PHONY: directories test_directories
