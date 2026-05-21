// =============================================================================
//  Strata::Onnx::OnnxGraph — minimal ONNX/protobuf wire-format reader
// -----------------------------------------------------------------------------
//  Decodes only what Phase 4 needs:
//    ModelProto.graph(7)
//      GraphProto.node(1)         -> NodeProto
//      GraphProto.initializer(5)  -> TensorProto (FLOAT, raw_data)
//      GraphProto.input(11)/output(12) -> ValueInfoProto.name(1)
//
//  Protobuf wire types: 0=varint, 1=i64, 2=length-delimited, 5=i32.
//  Unknown fields are skipped generically, so unrelated metadata in the file
//  (producer, doc strings, opset, ...) is tolerated.
// =============================================================================
#include "strata/onnx/onnx_graph.hpp"

#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>

static_assert(std::endian::native == std::endian::little,
              "OnnxGraph assumes little-endian raw_data (x86/x64)");

namespace Strata::Onnx {
namespace {

[[noreturn]] void bad(const char* msg) { throw std::runtime_error(msg); }

// Forward-only cursor over a byte span with bounds checking.
class Cursor {
public:
    explicit Cursor(std::span<const std::byte> b) : b_(b) {}

    [[nodiscard]] bool eof() const noexcept { return pos_ >= b_.size(); }

    std::uint64_t varint() {
        std::uint64_t v = 0;
        int shift = 0;
        while (true) {
            if (pos_ >= b_.size())            bad("varint past end");
            if (shift > 63)                   bad("varint too long");
            const std::uint8_t byte = static_cast<std::uint8_t>(b_[pos_++]);
            v |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) break;
            shift += 7;
        }
        return v;
    }

    // Returns {field_number, wire_type}.
    std::pair<std::uint32_t, std::uint32_t> tag() {
        const std::uint64_t t = varint();
        return {static_cast<std::uint32_t>(t >> 3),
                static_cast<std::uint32_t>(t & 0x7)};
    }

    std::span<const std::byte> length_delimited() {
        const std::uint64_t n = varint();
        if (n > b_.size() - pos_) bad("length-delimited past end");
        const auto s = b_.subspan(pos_, static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return s;
    }

    void skip(std::uint32_t wire_type) {
        switch (wire_type) {
            case 0: (void)varint();                       break;  // varint
            case 1: advance(8);                           break;  // i64
            case 2: (void)length_delimited();             break;  // bytes
            case 5: advance(4);                           break;  // i32
            default: bad("unknown wire type");
        }
    }

private:
    void advance(std::size_t n) {
        if (n > b_.size() - pos_) bad("fixed field past end");
        pos_ += n;
    }
    std::span<const std::byte> b_;
    std::size_t                pos_ = 0;
};

std::string to_string(std::span<const std::byte> s) {
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

// --- TensorProto (initializer) ---------------------------------------------
Initializer parse_tensor(std::span<const std::byte> buf) {
    Initializer t;
    std::span<const std::byte> raw;
    Cursor c(buf);
    while (!c.eof()) {
        const auto [field, wt] = c.tag();
        switch (field) {
            case 1:  // dims (repeated int64; packed or single)
                if (wt == 2) {
                    Cursor d(c.length_delimited());
                    while (!d.eof())
                        t.dims.push_back(static_cast<std::int64_t>(d.varint()));
                } else {
                    t.dims.push_back(static_cast<std::int64_t>(c.varint()));
                }
                break;
            case 8:  t.name = to_string(c.length_delimited()); break;  // name
            case 9:  raw = c.length_delimited(); break;               // raw_data
            default: c.skip(wt); break;  // data_type(2) etc. — unused here
        }
    }
    // raw_data is little-endian float32 (validated dtype upstream by usage).
    t.data = std::span<const float>(
        reinterpret_cast<const float*>(raw.data()),
        raw.size() / sizeof(float));
    return t;
}

// --- AttributeProto: pull the integer value when name == "transB" ----------
void parse_attribute(std::span<const std::byte> buf, Node& n) {
    std::string name;
    std::int64_t ival = 0;
    Cursor c(buf);
    while (!c.eof()) {
        const auto [field, wt] = c.tag();
        switch (field) {
            case 1: name = to_string(c.length_delimited()); break;  // name
            case 3: ival = static_cast<std::int64_t>(c.varint()); break;  // i
            default: c.skip(wt); break;
        }
    }
    if (name == "transB") n.trans_b = ival;
}

// --- NodeProto -------------------------------------------------------------
Node parse_node(std::span<const std::byte> buf) {
    Node n;
    Cursor c(buf);
    while (!c.eof()) {
        const auto [field, wt] = c.tag();
        switch (field) {
            case 1: n.input.push_back(to_string(c.length_delimited()));  break;
            case 2: n.output.push_back(to_string(c.length_delimited())); break;
            case 4: n.op_type = to_string(c.length_delimited());         break;
            case 5: parse_attribute(c.length_delimited(), n);            break;
            default: c.skip(wt); break;
        }
    }
    return n;
}

// --- ValueInfoProto: just the name -----------------------------------------
std::string parse_value_info_name(std::span<const std::byte> buf) {
    Cursor c(buf);
    while (!c.eof()) {
        const auto [field, wt] = c.tag();
        if (field == 1) return to_string(c.length_delimited());
        c.skip(wt);
    }
    return {};
}

} // namespace

OnnxGraph::OnnxGraph(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) bad("OnnxGraph: cannot open model file");
    const std::streamsize sz = f.tellg();
    if (sz <= 0) bad("OnnxGraph: empty model file");
    blob_.resize(static_cast<std::size_t>(sz));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(blob_.data()), sz);

    // ModelProto: find graph (field 7, length-delimited).
    std::span<const std::byte> graph;
    {
        Cursor c(blob_);
        while (!c.eof()) {
            const auto [field, wt] = c.tag();
            if (field == 7 && wt == 2) { graph = c.length_delimited(); break; }
            c.skip(wt);
        }
    }
    if (graph.empty()) bad("OnnxGraph: no GraphProto found");

    // GraphProto.
    Cursor c(graph);
    bool have_input = false, have_output = false;
    while (!c.eof()) {
        const auto [field, wt] = c.tag();
        switch (field) {
            case 1:  nodes_.push_back(parse_node(c.length_delimited())); break;
            case 5:  inits_.push_back(parse_tensor(c.length_delimited())); break;
            case 11: {
                std::string nm = parse_value_info_name(c.length_delimited());
                if (!have_input) { input_ = std::move(nm); have_input = true; }
                break;
            }
            case 12: {
                std::string nm = parse_value_info_name(c.length_delimited());
                if (!have_output) { output_ = std::move(nm); have_output = true; }
                break;
            }
            default: c.skip(wt); break;
        }
    }
    if (nodes_.empty())   bad("OnnxGraph: graph has no nodes");
    if (input_.empty())   bad("OnnxGraph: graph has no input");
    if (output_.empty())  bad("OnnxGraph: graph has no output");
}

const Initializer* OnnxGraph::find(std::string_view name) const noexcept {
    for (const auto& t : inits_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

} // namespace Strata::Onnx
