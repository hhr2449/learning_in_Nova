#include <iostream>
#include <future>
#include <thread>
#include <chrono>
int work() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 42;
}
int main() {
    // future用于获取一个异步操作的结果
    // 这里res是一个future<int>类型的对象，代表他会获取一个int类型的异步操作结果
    // async表示异步执行，launch::async表示新开一个线程执行work函数
    // future抽象了线程的创建和管理的细节，无需直接管理线程的生命周期和同步的细节，而是直接获取结果
    std::future<int> res = std::async(std::launch::async, work);
    std::cout << "waiting..." << std::endl;
    // get用于获取异步操作的结果，如果还没有完成，则会阻塞等待
    std::cout << "result: " << res.get() << std::endl;
    return 0;
}