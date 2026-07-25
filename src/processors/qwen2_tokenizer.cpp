#include "processors/qwen2_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace ncnn_omni {
namespace {

void append_utf8(std::string& out, uint32_t cp)
{
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

std::vector<std::string> utf8_chars(const std::string& text)
{
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        size_t count = 1;
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        if ((lead & 0xe0) == 0xc0) count = 2;
        else if ((lead & 0xf0) == 0xe0) count = 3;
        else if ((lead & 0xf8) == 0xf0) count = 4;
        if (i + count > text.size()) count = 1;
        chars.push_back(text.substr(i, count));
        i += count;
    }
    return chars;
}

uint32_t codepoint(const std::string& ch)
{
    const unsigned char* s = reinterpret_cast<const unsigned char*>(ch.data());
    if (ch.size() == 1) return s[0];
    if (ch.size() == 2) return ((s[0] & 0x1f) << 6) | (s[1] & 0x3f);
    if (ch.size() == 3) return ((s[0] & 0x0f) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f);
    if (ch.size() == 4)
        return ((s[0] & 0x07) << 18) | ((s[1] & 0x3f) << 12) | ((s[2] & 0x3f) << 6) |
               (s[3] & 0x3f);
    return 0xfffd;
}

bool is_newline(const std::string& ch) { return ch == "\n" || ch == "\r"; }

bool is_space(const std::string& ch)
{
    const uint32_t cp = codepoint(ch);
    return cp == 0x09 || cp == 0x0a || cp == 0x0b || cp == 0x0c || cp == 0x0d || cp == 0x20 ||
           cp == 0x85 || cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

bool is_number(const std::string& ch)
{
    const uint32_t cp = codepoint(ch);
    return (cp >= '0' && cp <= '9') || (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06f0 && cp <= 0x06f9) || (cp >= 0xff10 && cp <= 0xff19);
}

bool is_unicode_punctuation(uint32_t cp)
{
    return (cp >= 0x2000 && cp <= 0x206f) || (cp >= 0x2e00 && cp <= 0x2e7f) ||
           (cp >= 0x3000 && cp <= 0x303f) || (cp >= 0xfe10 && cp <= 0xfe1f) ||
           (cp >= 0xfe30 && cp <= 0xfe4f) || (cp >= 0xff00 && cp <= 0xff0f) ||
           (cp >= 0xff1a && cp <= 0xff20) || (cp >= 0xff3b && cp <= 0xff40) ||
           (cp >= 0xff5b && cp <= 0xff65);
}

bool is_letter(const std::string& ch)
{
    const uint32_t cp = codepoint(ch);
    if (cp < 128) return std::isalpha(static_cast<unsigned char>(cp)) != 0;
    if (is_space(ch) || is_number(ch) || is_unicode_punctuation(cp)) return false;
    // This covers the letter scripts supported by Qwen3-ASR. Symbols and emoji
    // fall through to punctuation ranges or are handled as non-letters.
    return (cp >= 0x00c0 && cp <= 0x02af) || (cp >= 0x0370 && cp <= 0x052f) ||
           (cp >= 0x0590 && cp <= 0x1fff) || (cp >= 0x3040 && cp <= 0x30ff) ||
           (cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xac00 && cp <= 0xd7af) ||
           (cp >= 0xf900 && cp <= 0xfaff);
}

std::string pair_key(const std::string& first, const std::string& second)
{
    return first + '\x1f' + second;
}

bool parse_hex4(const std::string& text, size_t& pos, uint32_t& value)
{
    if (pos + 4 > text.size()) return false;
    value = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = text[pos++];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= c - '0';
        else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
        else return false;
    }
    return true;
}

bool parse_json_string(const std::string& text, size_t& pos, std::string& value)
{
    if (pos >= text.size() || text[pos++] != '"') return false;
    value.clear();
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') return true;
        if (c != '\\') {
            value.push_back(c);
            continue;
        }
        if (pos >= text.size()) return false;
        const char escaped = text[pos++];
        switch (escaped) {
        case '"': value.push_back('"'); break;
        case '\\': value.push_back('\\'); break;
        case '/': value.push_back('/'); break;
        case 'b': value.push_back('\b'); break;
        case 'f': value.push_back('\f'); break;
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case 'u': {
            uint32_t cp = 0;
            if (!parse_hex4(text, pos, cp)) return false;
            if (cp >= 0xd800 && cp <= 0xdbff && pos + 6 <= text.size() && text[pos] == '\\' &&
                text[pos + 1] == 'u') {
                pos += 2;
                uint32_t low = 0;
                if (!parse_hex4(text, pos, low) || low < 0xdc00 || low > 0xdfff) return false;
                cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
            }
            append_utf8(value, cp);
            break;
        }
        default: return false;
        }
    }
    return false;
}

Result<std::unordered_map<std::string, int32_t>> read_vocab(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return Result<std::unordered_map<std::string, int32_t>>("cannot open vocab: " + path);
    std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    auto whitespace = [&]() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    };
    whitespace();
    if (pos >= text.size() || text[pos++] != '{')
        return Result<std::unordered_map<std::string, int32_t>>("invalid vocab JSON object");
    std::unordered_map<std::string, int32_t> vocab;
    while (true) {
        whitespace();
        if (pos < text.size() && text[pos] == '}') break;
        std::string key;
        if (!parse_json_string(text, pos, key))
            return Result<std::unordered_map<std::string, int32_t>>("invalid vocab JSON string");
        whitespace();
        if (pos >= text.size() || text[pos++] != ':')
            return Result<std::unordered_map<std::string, int32_t>>("invalid vocab JSON separator");
        whitespace();
        size_t end = pos;
        while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) ++end;
        if (end == pos) return Result<std::unordered_map<std::string, int32_t>>("invalid vocab token id");
        vocab.emplace(std::move(key), static_cast<int32_t>(std::stol(text.substr(pos, end - pos))));
        pos = end;
        whitespace();
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == '}') break;
        return Result<std::unordered_map<std::string, int32_t>>("invalid vocab JSON entry delimiter");
    }
    return Result<std::unordered_map<std::string, int32_t>>(std::move(vocab));
}

} // namespace

Result<bool> Qwen2Tokenizer::load(const std::string& vocab_json, const std::string& merges_txt)
{
    auto vocab = read_vocab(vocab_json);
    if (!vocab) return Result<bool>(vocab.error());
    encoder_ = std::move(vocab.value());
    decoder_.clear();
    for (const auto& item : encoder_) decoder_[item.second] = item.first;

    // GPT-2 reversible byte encoder used by Qwen2.
    std::vector<int> bytes;
    for (int i = '!'; i <= '~'; ++i) bytes.push_back(i);
    for (int i = 0xa1; i <= 0xac; ++i) bytes.push_back(i);
    for (int i = 0xae; i <= 0xff; ++i) bytes.push_back(i);
    std::vector<int> codepoints = bytes;
    int extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            codepoints.push_back(256 + extra++);
        }
    }
    byte_encoder_.clear();
    byte_decoder_.clear();
    for (size_t i = 0; i < bytes.size(); ++i) {
        std::string encoded;
        append_utf8(encoded, static_cast<uint32_t>(codepoints[i]));
        byte_encoder_[static_cast<unsigned char>(bytes[i])] = encoded;
        byte_decoder_[encoded] = static_cast<unsigned char>(bytes[i]);
    }

    std::ifstream merges(merges_txt);
    if (!merges) return Result<bool>("cannot open merges: " + merges_txt);
    merge_rank_.clear();
    std::string line;
    int rank = 0;
    while (std::getline(merges, line)) {
        if (line.empty() || line.rfind("#version:", 0) == 0) continue;
        std::istringstream parts(line);
        std::string first;
        std::string second;
        if (parts >> first >> second) merge_rank_[pair_key(first, second)] = rank++;
    }

    const std::vector<std::pair<int32_t, std::string>> added = {
        {151643, "<|endoftext|>"}, {151644, "<|im_start|>"}, {151645, "<|im_end|>"},
        {151669, "<|audio_start|>"}, {151670, "<|audio_end|>"}, {151676, "<|audio_pad|>"},
        {151704, "<asr_text>"}};
    for (const auto& item : added) {
        encoder_[item.second] = item.first;
        decoder_[item.first] = item.second;
    }
    bpe_cache_.clear();
    return Result<bool>(true);
}

std::vector<std::string> Qwen2Tokenizer::pretokenize(const std::string& text) const
{
    const auto chars = utf8_chars(text);
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < chars.size()) {
        const size_t begin = i;
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (chars[i] == "'") {
            static const std::vector<std::string> suffixes = {"re", "ve", "ll", "s", "t", "m", "d"};
            bool matched = false;
            for (const auto& suffix : suffixes) {
                if (i + 1 + suffix.size() > chars.size()) continue;
                std::string candidate;
                for (size_t j = 0; j < suffix.size(); ++j) candidate += chars[i + 1 + j];
                std::transform(candidate.begin(), candidate.end(), candidate.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (candidate == suffix) {
                    i += 1 + suffix.size();
                    matched = true;
                    break;
                }
            }
            if (matched) {
                std::string token;
                for (size_t j = begin; j < i; ++j) token += chars[j];
                tokens.push_back(std::move(token));
                continue;
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!is_newline(chars[i]) && !is_letter(chars[i]) && !is_number(chars[i]) &&
            i + 1 < chars.size() && is_letter(chars[i + 1]))
            ++i;
        if (i < chars.size() && is_letter(chars[i])) {
            while (i < chars.size() && is_letter(chars[i])) ++i;
        } else if (is_number(chars[i])) {
            ++i; // Qwen2 deliberately isolates each Unicode number.
        } else {
            // Optional one ASCII space, then punctuation/symbol run and newlines.
            if (chars[i] == " " && i + 1 < chars.size() && !is_space(chars[i + 1]) &&
                !is_letter(chars[i + 1]) && !is_number(chars[i + 1]))
                ++i;
            if (i < chars.size() && !is_space(chars[i]) && !is_letter(chars[i]) && !is_number(chars[i])) {
                while (i < chars.size() && !is_space(chars[i]) && !is_letter(chars[i]) && !is_number(chars[i])) ++i;
                while (i < chars.size() && is_newline(chars[i])) ++i;
            } else {
                // Covers \s*[\r\n]+, trailing whitespace, and ordinary whitespace.
                while (i < chars.size() && is_space(chars[i])) ++i;
            }
        }
        if (i == begin) ++i;
        std::string token;
        for (size_t j = begin; j < i; ++j) token += chars[j];
        tokens.push_back(std::move(token));
    }
    return tokens;
}

std::vector<std::string> Qwen2Tokenizer::bpe(const std::string& token)
{
    auto cached = bpe_cache_.find(token);
    if (cached != bpe_cache_.end()) return cached->second;
    std::vector<std::string> word = utf8_chars(token);
    while (word.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best = word.size();
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto rank = merge_rank_.find(pair_key(word[i], word[i + 1]));
            if (rank != merge_rank_.end() && rank->second < best_rank) {
                best_rank = rank->second;
                best = i;
            }
        }
        if (best == word.size()) break;
        const std::string first = word[best];
        const std::string second = word[best + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(word[i++]);
            }
        }
        word = std::move(merged);
    }
    bpe_cache_[token] = word;
    return word;
}

Result<std::vector<int32_t>> Qwen2Tokenizer::encode(const std::string& text)
{
    std::vector<int32_t> ids;
    for (const std::string& piece : pretokenize(text)) {
        std::string bytes;
        for (unsigned char byte : piece) bytes += byte_encoder_.at(byte);
        for (const std::string& symbol : bpe(bytes)) {
            auto found = encoder_.find(symbol);
            if (found == encoder_.end()) return Result<std::vector<int32_t>>("token is absent from Qwen vocab");
            ids.push_back(found->second);
        }
    }
    return Result<std::vector<int32_t>>(std::move(ids));
}

Result<std::string> Qwen2Tokenizer::decode(const std::vector<int32_t>& ids,
                                           bool skip_special_tokens) const
{
    static const std::unordered_set<int32_t> special = {
        151643, 151644, 151645, 151669, 151670, 151676};
    std::string encoded;
    for (int32_t id : ids) {
        if (skip_special_tokens && special.count(id)) continue;
        auto token = decoder_.find(id);
        if (token == decoder_.end())
            return Result<std::string>::failure("unknown token id: " + std::to_string(id));
        if (id >= 151643) {
            encoded += token->second;
            continue;
        }
        for (const std::string& ch : utf8_chars(token->second)) {
            auto byte = byte_decoder_.find(ch);
            if (byte == byte_decoder_.end())
                return Result<std::string>::failure("invalid byte-level token");
            encoded.push_back(static_cast<char>(byte->second));
        }
    }
    return Result<std::string>(std::move(encoded));
}

} // namespace ncnn_omni
