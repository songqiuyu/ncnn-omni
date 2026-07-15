#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ncnn_omni {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    template <typename U = T,
              typename std::enable_if<!std::is_same<U, std::string>::value, int>::type = 0>
    Result(std::string error) : error_(std::move(error)) {}

    static Result failure(std::string error)
    {
        Result result;
        result.error_ = std::move(error);
        return result;
    }

    explicit operator bool() const { return error_.empty(); }
    const std::string& error() const { return error_; }

    T& value()
    {
        if (!*this) throw Error(error_);
        return value_;
    }

    const T& value() const
    {
        if (!*this) throw Error(error_);
        return value_;
    }

private:
    Result() = default;
    T value_{};
    std::string error_;
};

} // namespace ncnn_omni
