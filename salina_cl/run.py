import time

import hydra
import torch
from salina import instantiate_class

from .logger import process_cfg, process_cfg_csp


@hydra.main(config_path="configs/", config_name="csp.yaml")
def main(cfg):
    _start = time.time()
    if "path" in cfg.model.params:
        if "csp" in cfg.name:
            cfg = process_cfg_csp(cfg)
        else:
            cfg = process_cfg(cfg)

    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg, verbose =False)
    model = instantiate_class(cfg.model)
    scenario = instantiate_class(cfg.scenario)
    #logger_evaluation = logger.get_logger("evaluation/")
    #logger_evaluation.logger.modulo = 1
    stage = model.get_stage()
    for train_task in scenario.train_tasks()[stage:]:
        model.train(train_task,logger)
        evaluation = model.evaluate(scenario.test_tasks(),logger)
        metrics = {}
        for tid in evaluation:
            for k,v in evaluation[tid].items():
                logger.add_scalar("evaluation/"+str(tid)+"_"+k,v,stage)
                metrics[k] = v + metrics.get(k,0)
        for k,v in metrics.items():
            logger.add_scalar("evaluation/aggregate_"+k,v / len(evaluation),stage)
        m_size = model.memory_size()
        for k,v in m_size.items():
            logger.add_scalar("memory/"+k,v,stage)
        stage+=1
    logger.close()
    logger.message("time elapsed: "+str(round((time.time()-_start),0))+" sec")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")
    main()