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
    // Accept success values only when the caller actually supplies T. Without
    // this constraint, Result<bool>("error") converts the string pointer to
    // true and silently reports success instead of selecting the error overload.
    template <typename U,
              typename std::enable_if<
                  std::is_same<typename std::decay<U>::type, T>::value, int>::type = 0>
    Result(U&& value) : value_(std::forward<U>(value)) {}
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
