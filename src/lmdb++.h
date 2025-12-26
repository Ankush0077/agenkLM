// src/lmdb++.h (Corrected for MDB_MAP_FULL)
#ifndef LMDB_PLUS_PLUS_H
#define LMDB_PLUS_PLUS_H

#include <lmdb.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>

namespace lmdb {

// Error handling
class exception : public std::runtime_error {
public:
    exception(const std::string& msg, int err)
        : std::runtime_error(msg + ": " + mdb_strerror(err)), code(err) {}
    int code;
};

// Environment class
class env {
private:
    MDB_env* mdb_env;
public:
    // THE FIX: Added a mapsize parameter with a large default (10 GiB)
    env(const char* path, MDB_dbi flags, mdb_mode_t mode, size_t mapsize = 10ULL * 1024 * 1024 * 1024) {
        if (auto rc = mdb_env_create(&mdb_env)) throw exception("mdb_env_create", rc);
        
        // Set the new map size before opening the environment
        if (auto rc = mdb_env_set_mapsize(mdb_env, mapsize)) throw exception("mdb_env_set_mapsize", rc);
        
        if (auto rc = mdb_env_set_maxdbs(mdb_env, 32)) throw exception("mdb_env_set_maxdbs", rc);
        if (auto rc = mdb_env_open(mdb_env, path, flags, mode)) throw exception("mdb_env_open", rc);
    }
    ~env() { if (mdb_env) mdb_env_close(mdb_env); }
    operator MDB_env*() { return mdb_env; }
};

// Transaction class
class txn {
private:
    MDB_txn* mdb_txn;
public:
    txn(MDB_env* env, MDB_txn* parent, MDB_dbi flags) {
        if (auto rc = mdb_txn_begin(env, parent, flags, &mdb_txn)) throw exception("mdb_txn_begin", rc);
    }
    ~txn() { if (mdb_txn) mdb_txn_commit(mdb_txn); } // auto-commit on destruction
    void abort() { mdb_txn_abort(mdb_txn); mdb_txn = nullptr; }
    operator MDB_txn*() { return mdb_txn; }
};

// Database class
class dbi {
private:
    MDB_dbi mdb_dbi;
public:
    dbi(MDB_txn* txn, const char* name, MDB_dbi flags) {
        if (auto rc = mdb_dbi_open(txn, name, flags, &mdb_dbi)) throw exception("mdb_dbi_open", rc);
    }
    operator MDB_dbi() { return mdb_dbi; }
};

// Data view class
struct val {
    MDB_val mdb_val;
    val() : mdb_val{0, nullptr} {}
    val(size_t size, void* data) : mdb_val{size, data} {}
    template<typename T> val(const T& t) : mdb_val{sizeof(t), (void*)&t} {}
    template<typename T> val(const std::vector<T>& v) : mdb_val{v.size() * sizeof(T), (void*)v.data()} {}
    val(const std::string& s) : mdb_val{s.size(), (void*)s.data()} {}

    template<typename T> T as() const {
        T t;
        if (mdb_val.mv_size != sizeof(T)) throw std::runtime_error("invalid size for type");
        std::memcpy(&t, mdb_val.mv_data, sizeof(T));
        return t;
    }
    std::string as_string() const { return std::string((char*)mdb_val.mv_data, mdb_val.mv_size); }
};

// Put operation
inline void put(MDB_txn* txn, MDB_dbi dbi, const val& key, const val& data, MDB_dbi flags = 0) {
    if (auto rc = mdb_put(txn, dbi, const_cast<MDB_val*>(&key.mdb_val), const_cast<MDB_val*>(&data.mdb_val), flags)) 
        throw exception("mdb_put", rc);
}

} // namespace lmdb

#endif // LMDB_PLUS_PLUS_H