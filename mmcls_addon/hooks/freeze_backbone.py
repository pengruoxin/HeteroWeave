from mmcv.runner import HOOKS, Hook, load_checkpoint


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Load and freeze a classifier backbone for head-only transfer."""

    def __init__(
        self,
        checkpoint=None,
        train_head_only=True,
        freeze_backbone=True,
        revise_keys=(
            (r'^module\.backbone\.', ''),
            (r'^backbone\.', ''),
        ),
    ):
        self.checkpoint = checkpoint
        self.train_head_only = train_head_only
        self.freeze_backbone = freeze_backbone
        self.revise_keys = revise_keys
        self._loaded = False

    def _apply_freeze(self, runner):
        model = _unwrap_model(runner.model)
        if self.checkpoint and not self._loaded:
            load_checkpoint(
                model.backbone,
                self.checkpoint,
                map_location='cpu',
                strict=False,
                revise_keys=self.revise_keys)
            self._loaded = True

        if self.train_head_only:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        elif self.freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False

        if self.train_head_only or self.freeze_backbone:
            model.backbone.eval()

    def before_run(self, runner):
        self._apply_freeze(runner)
        model = _unwrap_model(runner.model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        runner.logger.info(
            'FreezeBackboneHook trainable params: %d / %d', trainable, total)

    def before_train_epoch(self, runner):
        self._apply_freeze(runner)

    def before_train_iter(self, runner):
        if self.train_head_only or self.freeze_backbone:
            model = _unwrap_model(runner.model)
            model.backbone.eval()
