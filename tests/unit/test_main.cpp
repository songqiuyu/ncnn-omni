#include <exception>
#include <iostream>

void test_whisper_log_mel();
void test_qwen2_tokenizer();

int main()
{
    try {
        test_whisper_log_mel();
        test_qwen2_tokenizer();
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
